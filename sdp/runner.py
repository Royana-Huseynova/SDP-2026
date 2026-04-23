"""
Train / inference runners.

The AllClear codebase exposes its full pipeline as a CLI module
(`python -m allclear.benchmark`). Re-implementing it here would duplicate
hundreds of lines of model glue, so the runner *invokes the existing
benchmark* in a subprocess with the right arguments. This keeps the
unified API thin and avoids drift from upstream.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import config
from datasets import DatasetHandle


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _ensure_kmp_workaround(env: dict[str, str]) -> dict[str, str]:
    """Avoid the 'OMP: libiomp5md.dll already loaded' crash on Windows."""
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def _resolve_run_dir(
    handle: DatasetHandle,
    experiment_output_path: Path,
    model_name: str,
    exp_name: str,
) -> Path:
    """
    Mirror the layout that allclear.benchmark uses so visualize/metrics
    can find prediction files without extra arguments.
    """
    json_stem = (
        Path(handle.dataset_fpath).stem
        if handle.dataset_fpath else "run"
    )
    return Path(experiment_output_path) / model_name / exp_name / "AllClear" / json_stem


# --------------------------------------------------------------------- #
# AllClear benchmark runner
# --------------------------------------------------------------------- #
def _run_allclear_benchmark(
    handle: DatasetHandle,
    *,
    model_name: str = "uncrtaints",
    device: str = "cpu",
    batch_size: int = 1,
    num_workers: int = 0,
    draw_vis: int = 0,
    experiment_output_path: Optional[Path] = None,
    uc_baseline_base_path: Path = Path("baselines/UnCRtainTS/model"),
    uc_weight_folder: Path = Path("checkpoints"),
    uc_exp_name: str = "utae",
    extra_args: Optional[list[str]] = None,
    dry_run: bool = False,
) -> Path:
    """Invoke `python -m allclear.benchmark` and return the run_dir."""
    if handle.dataset_fpath is None:
        raise ValueError("Handle has no dataset_fpath; cannot launch benchmark.")

    if experiment_output_path is None:
        experiment_output_path = config.RESULTS_DIR / "baseline" / model_name / uc_exp_name

    cmd: list[str] = [
        sys.executable, "-m", "allclear.benchmark",
        "--dataset-fpath", str(handle.dataset_fpath),
        "--model-name", model_name,
        "--device", device,
        "--main-sensor", str(handle.extras.get("main_sensor", "s2_toa")),
        "--target-mode", str(handle.extras.get("target_mode", "s2p")),
        "--tx", str(handle.extras.get("tx", 3)),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--draw-vis", str(draw_vis),
        "--experiment-output-path", str(experiment_output_path),
        "--uc-baseline-base-path", str(uc_baseline_base_path),
        "--uc-weight-folder", str(uc_weight_folder),
        "--uc-exp-name", uc_exp_name,
    ]
    aux_sensors = handle.extras.get("aux_sensors") or []
    if aux_sensors:
        cmd += ["--aux-sensors", *aux_sensors]
    aux_data = handle.extras.get("aux_data") or []
    if aux_data:
        cmd += ["--aux-data", *aux_data]
    if extra_args:
        cmd += list(extra_args)

    env = _ensure_kmp_workaround(os.environ.copy())

    sys.stderr.write(
        f"[sdp] Launching from {config.EXTERNAL_DIR}:\n"
        f"      {' '.join(shlex.quote(c) for c in cmd)}\n"
    )

    if dry_run:
        return _resolve_run_dir(handle, experiment_output_path, model_name, uc_exp_name)

    subprocess.run(cmd, cwd=str(config.EXTERNAL_DIR), env=env, check=True)
    return _resolve_run_dir(handle, experiment_output_path, model_name, uc_exp_name)


# --------------------------------------------------------------------- #
# Proba-V runner stub
# --------------------------------------------------------------------- #
def _run_probav(handle: DatasetHandle, **kwargs: Any) -> Path:
    raise NotImplementedError(
        "Proba-V training/inference is not wired yet. Implement "
        "sdp/_probav.py and the matching train loop, then plug into runner.py."
    )


# --------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------- #
def train(handle: DatasetHandle, **kwargs: Any) -> DatasetHandle:
    """
    Run training / benchmark for the given handle.

    AllClear's benchmark is inference-only by design; calling `train` on
    an AllClear handle delegates to the benchmark engine which evaluates
    pretrained weights. For Proba-V, this is where a real training loop
    belongs.
    """
    if handle.name == "allclear":
        model_name = kwargs.pop("model_name", None) or handle.variant or "uncrtaints"
        run_dir = _run_allclear_benchmark(handle, model_name=model_name, **kwargs)
        handle.run_dir = run_dir
        handle.model_name = model_name
        return handle
    elif handle.name == "probav":
        _run_probav(handle, **kwargs)
        return handle
    else:
        raise ValueError(f"Unsupported dataset {handle.name!r}")


def inference(handle: DatasetHandle, **kwargs: Any) -> DatasetHandle:
    """
    Same as `train` for AllClear (the benchmark *is* inference).
    Kept as a separate verb so the API matches the user's notes.
    """
    return train(handle, **kwargs)
