import os
from pathlib import Path

VPS_URL = os.environ.get("VPS_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
WORKER_ID = os.environ.get("WORKER_ID", "")
WORKER_NAME = os.environ.get("WORKER_NAME", "colab-t4")
LEASE_SAFETY_SECONDS = int(os.environ.get("LEASE_SAFETY_SECONDS", "60"))
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "15"))

COMFY_OUTPUT_DIR = Path("/content/ComfyUI/output")
INPUT_DIR = Path("/content/ComfyUI/input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
COMFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = Path("/content/worker_state.json")
