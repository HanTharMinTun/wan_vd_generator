"""Colab worker loop for the WAN distributed VPS server."""

import json
import time
import traceback
import threading
from pathlib import Path

import httpx

from . import config
from . import wan_adapter


def _headers() -> dict:
    return {
        "X-Worker-Id": config.WORKER_ID,
        "X-Worker-Token": config.WORKER_TOKEN,
    }


def _client() -> httpx.Client:
    if not config.VPS_URL:
        raise RuntimeError("VPS_URL is not configured")
    return httpx.Client(base_url=config.VPS_URL, timeout=60.0)


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _persist_state() -> None:
    try:
        config.STATE_FILE.write_text(json.dumps({
            "worker_id": config.WORKER_ID,
            "token": config.WORKER_TOKEN,
            "name": config.WORKER_NAME,
        }))
    except Exception:
        pass


def _load_state() -> None:
    try:
        if config.STATE_FILE.is_file():
            state = json.loads(config.STATE_FILE.read_text())
            config.WORKER_ID = state.get("worker_id", config.WORKER_ID)
            config.WORKER_TOKEN = state.get("token", config.WORKER_TOKEN)
            config.WORKER_NAME = state.get("name", config.WORKER_NAME)
    except Exception:
        pass


def register_once(client: httpx.Client, name: str) -> tuple[int, str]:
    response = client.post(
        "/api/workers/register",
        json={"name": name, "gpu_info": _gpu_name()},
    )
    response.raise_for_status()
    data = response.json()
    config.WORKER_ID = str(data["worker_id"])
    config.WORKER_TOKEN = data["token"]
    config.WORKER_NAME = data["name"]
    _persist_state()
    print(f"[worker] registered id={config.WORKER_ID} name={config.WORKER_NAME}")
    return int(config.WORKER_ID), config.WORKER_TOKEN


def ensure_registered(client: httpx.Client) -> None:
    if not (config.WORKER_ID and config.WORKER_TOKEN):
        register_once(client, config.WORKER_NAME or f"colab-{int(time.time())}")
        return
    try:
        response = client.get("/api/workers/me", headers=_headers())
        if response.status_code == 401:
            register_once(client, config.WORKER_NAME or f"colab-{int(time.time())}")
        else:
            response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[worker] identity check failed ({exc}); continuing")


def heartbeat(client: httpx.Client, job_id: int | None = None) -> None:
    try:
        client.post(
            "/api/workers/heartbeat",
            json={"current_job_id": job_id},
            headers=_headers(),
        )
    except httpx.HTTPError:
        pass


def run_forever() -> None:
    _load_state()
    with _client() as client:
        ensure_registered(client)
        print(f"[worker] polling {config.VPS_URL}")
        while True:
            try:
                _tick(client)
            except KeyboardInterrupt:
                print("[worker] stopped by user")
                return
            except Exception:
                print("[worker] unexpected error, continuing in 10s:")
                traceback.print_exc()
                time.sleep(10)


def _tick(client: httpx.Client) -> None:
    heartbeat(client)
    response = client.get("/api/worker/next-job", headers=_headers())
    if response.status_code == 401:
        ensure_registered(client)
        return
    response.raise_for_status()
    job = response.json()
    if not job:
        time.sleep(config.POLL_INTERVAL_SECONDS)
        return

    job_id = job["job_id"]
    print(f"[worker] claimed job {job_id} scene={job['scene_name']}")
    heartbeat(client, job_id)

    try:
        result_path = _process_job(client, job)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[worker] job {job_id} failed: {message}")
        _safe_fail(client, job_id, message)
        _free_cuda()
        return

    try:
        _upload(client, job_id, result_path)
        _safe_complete(client, job_id)
        print(f"[worker] job {job_id} COMPLETED -> {result_path}")
    except Exception as exc:
        print(f"[worker] upload/complete failed for {job_id}: {exc}")
        _safe_fail(client, job_id, f"upload: {exc}")
    finally:
        _free_cuda()


def _process_job(client: httpx.Client, job: dict) -> str:
    image_url = job["image_url"]
    if image_url.startswith("/"):
        image_url = config.VPS_URL + image_url

    response = client.get(image_url, timeout=120.0)
    response.raise_for_status()

    lower_url = job["image_url"].lower()
    suffix = ".jpg" if lower_url.endswith((".jpg", ".jpeg")) else ".webp" if lower_url.endswith(".webp") else ".png"
    image_path = config.INPUT_DIR / f"job_{job['job_id']}{suffix}"
    image_path.write_bytes(response.content)

    _report_progress(client, job["job_id"], 10)
    print(f"[worker] generating: {job['prompt'][:60]}...")
    _report_progress(client, job["job_id"], 25)

    stop_heartbeat = threading.Event()

    def _periodic_heartbeat():
        while not stop_heartbeat.wait(30):
            heartbeat(client, job["job_id"])

    heartbeat_thread = threading.Thread(target=_periodic_heartbeat, daemon=True)
    heartbeat_thread.start()

    heartbeat(client, job["job_id"])
    try:
        output = wan_adapter.generate(
            image_path=str(image_path),
            prompt=job["prompt"],
            settings=job.get("settings", {}),
        )
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)
        _report_progress(client, job["job_id"], 90)
        heartbeat(client, job["job_id"])
    return output


def _report_progress(client: httpx.Client, job_id: int, percent: int) -> None:
    try:
        client.post(
            f"/api/jobs/{job_id}/progress",
            json={"progress": percent},
            headers=_headers(),
            timeout=15.0,
        )
    except httpx.HTTPError:
        pass


def _upload(client: httpx.Client, job_id: int, path: str) -> None:
    with open(path, "rb") as output:
        response = client.post(
            f"/api/jobs/{job_id}/upload",
            files={"file": (Path(path).name, output, "video/mp4")},
            headers=_headers(),
            timeout=600.0,
        )
    response.raise_for_status()


def _safe_complete(client: httpx.Client, job_id: int) -> None:
    for attempt in range(3):
        try:
            response = client.post(
                f"/api/jobs/{job_id}/complete",
                headers=_headers(),
                timeout=30.0,
            )
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("could not report completion")


def _safe_fail(client: httpx.Client, job_id: int, message: str) -> None:
    for attempt in range(3):
        try:
            client.post(
                f"/api/jobs/{job_id}/fail",
                json={"error_message": message[:4000]},
                headers=_headers(),
                timeout=30.0,
            )
            return
        except httpx.HTTPError:
            time.sleep(2 ** attempt)


def _free_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
