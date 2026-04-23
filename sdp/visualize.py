"""
Visualization wrapper.

Delegates to `src/visualize_allclear.py` (executed as a module) so we
don't duplicate the figure-rendering code that already exists.
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


def visualize(
    handle: DatasetHandle,
    *,
    run_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    model: Optional[str] = None,
    num: int = 20,
    start: int = 0,
    out: Optional[Path] = None,
    no_stretch: bool = False,
    dry_run: bool = False,
) -> Path:
    """
    Render side-by-side panels for ``num`` samples from the given run.

    AllClear: input / prediction / target / diff / mask panels via
    ``src/visualize_allclear.py``.

    Proba-V: best-LR / SR-prediction / HR-target / diff panels rendered
    directly from ``sdp/_probav.py``.

    Parameters
    ----------
    handle : DatasetHandle
    run_dir : Path, optional
        Override the run_dir stored on the handle.
    json_path : Path, optional
        AllClear only — dataset JSON used for the run.
    model : str, optional
        AllClear only — model name.
    num : int
        Number of samples to render.
    start : int
        AllClear only — first sample index.
    out : Path, optional
        Output directory for PNGs.
    no_stretch : bool
        AllClear only — disable RGB histogram stretching.
    dry_run : bool
        Print resolved command without executing.
    """
    if handle.name == "probav":
        run_dir = Path(run_dir or handle.run_dir or "")
        if not run_dir or str(run_dir) == ".":
            raise ValueError(
                "No run_dir available. Call sdp.train()/inference() first, "
                "or pass run_dir explicitly."
            )
        if dry_run:
            return (run_dir / "vis")
        from ._probav import visualize_probav
        return visualize_probav(
            handle,
            run_dir=run_dir,
            num=num,
            out=Path(out) if out else None,
        )

    if handle.name != "allclear":
        raise NotImplementedError(f"visualize() does not support dataset {handle.name!r}.")

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

    cmd: list[str] = [
        sys.executable, "-m", "src.visualize_allclear",
        "--run_dir", str(run_dir),
        "--model", model,
        "--json", str(json_path),
        "--num", str(num),
        "--start", str(start),
    ]
    if out is not None:
        cmd += ["--out", str(out)]
    if no_stretch:
        cmd += ["--no_stretch"]

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    sys.stderr.write(
        f"[sdp] Visualization: cwd={config.REPO_ROOT}\n"
        f"      {' '.join(shlex.quote(c) for c in cmd)}\n"
    )
    if dry_run:
        return run_dir / "vis_json"

    subprocess.run(cmd, cwd=str(config.REPO_ROOT), env=env, check=True)
    return run_dir / "vis_json"
