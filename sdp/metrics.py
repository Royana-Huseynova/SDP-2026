"""
Metrics wrapper.

Calls `src/Metrics.py` (renamed from notes' `metrics.py`) as a module so
the same computation runs whether you trigger it from the CLI or the
Python API.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional
from . import config
from .datasets import DatasetHandle


def metrics(
    handle: DatasetHandle,
    *,
    run_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    model: Optional[str] = None,
    out_prefix: Optional[str] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> Path:
    """Compute MAE / RMSE / PSNR / SAM / SSIM / LPIPS / FID for a run."""
    if handle.name != "allclear":
        raise NotImplementedError(
            f"metrics() currently only supports AllClear (got {handle.name!r})."
        )

    run_dir = Path(run_dir or handle.run_dir or "")
    if not run_dir or str(run_dir) == ".":
        raise ValueError(
            "No run_dir available. Call sdp.train()/inference() first, "
            "or pass run_dir explicitly."
        )
    json_path = Path(json_path or handle.dataset_fpath or "")
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
    model = model or handle.model_name
    if model is None:
        raise ValueError("Model name not set; pass model=... or run train() first.")

    # src/Metrics.py exposes a CLI; invoke it via -m src.Metrics. (Use the
    # actual filename casing: Metrics.py.)
    cmd: list[str] = [
        sys.executable, str(config.SRC_DIR / "Metrics.py"),
        "--run_dir", str(run_dir),
        "--model", model,
        "--json", str(json_path),
    ]
    if out_prefix:
        cmd += ["--out_prefix", out_prefix]
    if device:
        cmd += ["--device", device]

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    sys.stderr.write(
        f"[sdp] Metrics: cwd={config.REPO_ROOT}\n"
        f"      {' '.join(shlex.quote(c) for c in cmd)}\n"
    )
    if dry_run:
        return run_dir

    subprocess.run(cmd, cwd=str(config.REPO_ROOT), env=env, check=True)
    return run_dir
