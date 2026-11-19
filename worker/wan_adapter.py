"""Thin adapter around the generate_video function from the Colab notebook."""

import time
from pathlib import Path


class WanNotReady(RuntimeError):
    pass


def _caller_globals() -> dict:
    import __main__

    return __main__.__dict__


def _ensure_notebook_ready() -> dict:
    g = _caller_globals()
    missing = [name for name in ("generate_video", "torch") if name not in g]
    if missing:
        raise WanNotReady(
            "Run the notebook's Prepare Environment cell first. Missing: "
            + ", ".join(missing)
        )
    return g


def generate(image_path: str, prompt: str, settings: dict, progress_cb=None) -> str:
    g = _ensure_notebook_ready()

    g.setdefault("clear_output", lambda *args, **kwargs: None)
    g.setdefault("display", lambda *args, **kwargs: None)

    allowed = {
        "width", "height", "frames", "fps", "seed", "steps", "cfg_scale",
        "sampler_name", "scheduler", "output_format", "overwrite",
        "positive_prompt", "negative_prompt", "prompt_assist",
        "use_lora", "LoRA_Strength", "use_lora2", "LoRA_Strength2",
        "use_lora3", "LoRA_Strength3", "use_lightx2v", "lightx2v_Strength",
        "lightx2v_steps", "use_pusa", "pusa_Strength", "pusa_steps",
        "use_sage_attention", "enable_flow_shift", "shift",
        "enable_flow_shift2", "shift2", "end_step1",
        "rel_l1_thresh", "start_percent", "end_percent",
    }
    kwargs = {"image_path": image_path, "positive_prompt": prompt}

    for key, value in (settings or {}).items():
        if key in allowed:
            kwargs[key] = value

    try:
        import sageattention  # noqa: F401
    except Exception:
        kwargs["use_sage_attention"] = False

    started = time.time()
    g["generate_video"](**kwargs)
    elapsed = time.time() - started

    output = g.get("output_path")
    if not output or not Path(output).is_file():
        raise RuntimeError(
            f"generate_video() finished in {elapsed:.1f}s but output_path "
            f"is missing or not a file: {output!r}"
        )
    return str(output)
