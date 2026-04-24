"""
PyTorch Dataset and built-in baseline for the ESA Proba-V super-resolution challenge.

On-disk layout (standard challenge structure):
    <root>/imgsetXXXXXX/
        LR000.png ... LR<N>.png   (128x128, uint16, 14-bit range 0-16383)
        QM000.png ... QM<N>.png   (128x128 binary clearance mask)
        HR.png                    (384x384, train split only)
        SM.png                    (384x384 binary clearance mask, train only)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .datasets import DatasetHandle

_MAX_VAL = 16383.0  # 14-bit Proba-V sensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProbaVDataset(Dataset):
    """
    Each item is one scene (imgsetXXXXXX/) and returns:

        lrs:   (T, 1, 128, 128) float32 [0,1]
        qms:   (T, 1, 128, 128) float32 binary  (1 = clear)
        scene: str
        hr:    (1, 384, 384) float32 [0,1]       — train split only
        sm:    (1, 384, 384) float32 binary       — train split only
    """

    def __init__(self, root: Path, scale: int = 3, split: str = "train") -> None:
        self.root = Path(root)
        self.scale = scale
        self.split = split
        self._scenes: list[Path] = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("imgset")
        )
        if not self._scenes:
            raise FileNotFoundError(f"No imgset* directories found in {self.root}")

    def __len__(self) -> int:
        return len(self._scenes)

    @staticmethod
    def _load_gray(path: Path) -> np.ndarray:
        """Load a (possibly 16-bit) grayscale PNG; returns (1, H, W) float32 in [0, 1]."""
        from PIL import Image
        arr = np.array(Image.open(path), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr / _MAX_VAL)[None]  # (1, H, W)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        """Load clearance mask PNG; returns (1, H, W) float32 binary (1=clear)."""
        from PIL import Image
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        return (arr > 0).astype(np.float32)[None]  # (1, H, W)

    def __getitem__(self, idx: int) -> dict:
        scene_dir = self._scenes[idx]
        lr_paths = sorted(scene_dir.glob("LR*.png"))
        qm_paths = sorted(scene_dir.glob("QM*.png"))

        if not lr_paths:
            raise FileNotFoundError(f"No LR*.png in {scene_dir}")

        lrs = np.stack([self._load_gray(p) for p in lr_paths])
        qms = (
            np.stack([self._load_mask(p) for p in qm_paths])
            if qm_paths else np.ones_like(lrs)
        )

        sample: dict = {
            "lrs": torch.from_numpy(lrs),
            "qms": torch.from_numpy(qms),
            "scene": scene_dir.name,
        }

        hr_path = scene_dir / "HR.png"
        if hr_path.exists():
            sample["hr"] = torch.from_numpy(self._load_gray(hr_path))
            sm_path = scene_dir / "SM.png"
            if sm_path.exists():
                sample["sm"] = torch.from_numpy(self._load_mask(sm_path))
            else:
                _, h, w = sample["hr"].shape
                sample["sm"] = torch.ones(1, h, w)

        return sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dataset(handle: "DatasetHandle") -> ProbaVDataset:
    """Return the dataset from the handle, constructing it if necessary."""
    if handle.torch_dataset is not None:
        return handle.torch_dataset
    root = handle.data_path / "probav" / handle.split / handle.extras.get("band", "NIR")
    return ProbaVDataset(root=root, scale=handle.extras.get("scale", 3), split=handle.split)


def _baseline_sr(lrs: torch.Tensor, qms: torch.Tensor, scale: int = 3) -> torch.Tensor:
    """
    Pixel-wise median composite of clear frames, bicubic ×scale upsample.

    Clear pixels across T frames are median-combined; cloudy-only pixels
    fall back to the frame-wise median. Result is then bicubic-upsampled.

    lrs : (T, 1, H, W) float32 [0,1]
    qms : (T, 1, H, W) float32 binary
    returns (1, H*scale, W*scale) float32 [0,1]
    """
    import torch.nn.functional as F

    T, C, H, W = lrs.shape

    # Mask cloudy pixels with NaN so nanmedian ignores them.
    masked = lrs.clone().squeeze(1).reshape(T, H * W)  # (T, H*W)
    qm_flat = qms.squeeze(1).reshape(T, H * W)         # (T, H*W)
    masked[qm_flat == 0] = float("nan")

    # Compute nanmedian across time axis.
    try:
        med = torch.nanmedian(masked, dim=0).values  # (H*W,)
    except AttributeError:
        # torch < 1.10 fallback — slow but correct
        med = torch.zeros(H * W)
        plain_med = lrs.squeeze(1).reshape(T, H * W).median(dim=0).values
        for i in range(H * W):
            col = masked[:, i]
            valid = col[~torch.isnan(col)]
            med[i] = valid.median() if valid.numel() > 0 else plain_med[i]

    # Fill pixels with no clear frame using per-pixel frame median.
    nan_mask = torch.isnan(med)
    if nan_mask.any():
        fallback = lrs.squeeze(1).reshape(T, H * W).median(dim=0).values
        med[nan_mask] = fallback[nan_mask]

    composite = med.reshape(1, 1, H, W).clamp(0.0, 1.0)
    sr = F.interpolate(composite, size=(H * scale, W * scale), mode="bicubic", align_corners=False)
    return sr.squeeze(0).clamp(0.0, 1.0)  # (1, H*scale, W*scale)


def _cpsnr(sr: torch.Tensor, hr: torch.Tensor, sm: Optional[torch.Tensor] = None) -> float:
    """Simple clearance-masked PSNR — fast version used during baseline inference."""
    mask = (sm.bool().squeeze() if sm is not None else torch.ones_like(hr, dtype=torch.bool).squeeze())
    if mask.sum() == 0:
        return float("nan")
    diff = (sr.squeeze() - hr.squeeze())[mask]
    mse = (diff ** 2).mean().item()
    return -10.0 * np.log10(mse) if mse > 0 else float("inf")


def _psnr_np(sr: np.ndarray, hr: np.ndarray) -> float:
    """Plain PSNR (dB) on float [0,1] arrays."""
    mse = float(np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _cpsnr_shifted(sr: np.ndarray, hr: np.ndarray, mask: np.ndarray, border: int = 3) -> float:
    """
    Official-style Proba-V cPSNR with sub-pixel shift search and bias correction.

    Searches all (u, v) shifts within ±border to find the best SR/HR alignment,
    applies per-shift brightness bias correction, and returns the highest cPSNR.
    """
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    mask = mask.astype(bool)
    H, W = sr.shape
    c = border
    sr_crop = sr[c:H - c, c:W - c].ravel()

    best_cMSE = np.inf
    for u in range(2 * c + 1):
        for v in range(2 * c + 1):
            hr_crop = hr[u:u + (H - 2 * c), v:v + (W - 2 * c)].ravel()
            m_crop = mask[u:u + (H - 2 * c), v:v + (W - 2 * c)].ravel()
            if m_crop.sum() == 0:
                continue
            _hr = hr_crop[m_crop]
            _sr = sr_crop[m_crop]
            b = np.mean(_hr - _sr)
            diff = _hr - (_sr + b)
            cMSE = float(np.mean(diff * diff))
            best_cMSE = min(best_cMSE, cMSE)

    if best_cMSE == np.inf:
        return float("nan")
    if best_cMSE == 0:
        return float("inf")
    return float(-10.0 * np.log10(best_cMSE))


def _sam_np(sr: np.ndarray, hr: np.ndarray) -> float:
    """Spectral Angle Mapper for single-band images (radians, lower is better)."""
    sr = sr.astype(np.float64).ravel()
    hr = hr.astype(np.float64).ravel()
    dot = np.dot(sr, hr)
    ns, nh = np.linalg.norm(sr), np.linalg.norm(hr)
    if ns == 0 or nh == 0:
        return 0.0
    return float(np.arccos(np.clip(dot / (ns * nh), -1.0, 1.0)))


def _ergas_np(sr: np.ndarray, hr: np.ndarray, scale: int = 3) -> float:
    """ERGAS — normalized global error for satellite SR (lower is better)."""
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    hr_mean = hr.mean()
    if hr_mean == 0:
        return 0.0
    rmse = float(np.sqrt(np.mean((sr - hr) ** 2)))
    return float(100.0 * (1.0 / scale) * (rmse / hr_mean))


def _save_sr_png(sr: torch.Tensor, path: Path) -> None:
    """Save (1, H, W) float32 [0,1] tensor as uint16 PNG."""
    from PIL import Image
    arr = (sr.squeeze(0).numpy() * _MAX_VAL).clip(0, _MAX_VAL).astype(np.uint16)
    img = Image.fromarray(arr)
    img.save(path)


def _load_sr_png(path: Path) -> torch.Tensor:
    """Load a uint16 SR PNG and return (1, H, W) float32 [0,1]."""
    from PIL import Image
    arr = np.array(Image.open(path), dtype=np.float32) / _MAX_VAL
    return torch.from_numpy(arr)[None]


# ---------------------------------------------------------------------------
# Public: baseline runner
# ---------------------------------------------------------------------------

def run_probav_baseline(handle: "DatasetHandle", *, run_dir: Path, **_: Any) -> None:
    """
    Run the clear-pixel-median-composite baseline over every scene.

    Outputs per scene:
        <run_dir>/<scene>/SR.png
    Outputs aggregate:
        <run_dir>/summary.json  (per-scene cPSNR + mean, when ground truth exists)
    """
    ds = _get_dataset(handle)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    n = len(ds)
    for i, sample in enumerate(ds):
        scene = sample["scene"]
        sr = _baseline_sr(sample["lrs"], sample["qms"], scale=ds.scale)

        scene_dir = run_dir / scene
        scene_dir.mkdir(exist_ok=True)
        _save_sr_png(sr, scene_dir / "SR.png")

        entry: dict = {"scene": scene}
        if "hr" in sample:
            entry["cpsnr"] = _cpsnr(sr, sample["hr"], sample.get("sm"))
        results.append(entry)

        sys.stderr.write(
            f"\r[sdp] probav_baseline {i + 1}/{n}: {scene}"
            + (f"  cPSNR={entry['cpsnr']:.3f} dB" if "cpsnr" in entry else "")
        )
        sys.stderr.flush()

    sys.stderr.write("\n")

    summary: dict[str, Any] = {"model": "probav_baseline", "scenes": results}
    cpsnr_vals = [r["cpsnr"] for r in results if "cpsnr" in r and not np.isnan(r["cpsnr"])]
    if cpsnr_vals:
        summary["mean_cpsnr"] = float(np.mean(cpsnr_vals))
        sys.stderr.write(f"[sdp] mean cPSNR = {summary['mean_cpsnr']:.4f} dB\n")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.stderr.write(f"[sdp] results saved to {run_dir}\n")


# ---------------------------------------------------------------------------
# Public: metrics
# ---------------------------------------------------------------------------

def compute_probav_metrics(handle: "DatasetHandle", *, run_dir: Path) -> dict:
    """
    Load saved SR predictions and compute cPSNR / PSNR / SSIM / RMSE / SAM / ERGAS
    against HR ground truth. Writes <run_dir>/metrics.json and returns the aggregate.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        _has_skimage = True
    except ImportError:
        _has_skimage = False

    ds = _get_dataset(handle)
    scale = ds.scale
    run_dir = Path(run_dir)
    results: list[dict] = []

    for sample in ds:
        scene = sample["scene"]
        sr_path = run_dir / scene / "SR.png"

        if not sr_path.exists():
            sys.stderr.write(f"[sdp] WARNING: no SR.png for {scene}, skipping\n")
            continue
        if "hr" not in sample:
            continue  # test split — no ground truth

        sr = _load_sr_png(sr_path)
        hr = sample["hr"]
        sm = sample.get("sm")

        sr_np = sr.squeeze().numpy().astype(np.float64)
        hr_np = hr.squeeze().numpy().astype(np.float64)
        mask_np = (sm.squeeze().numpy() > 0) if sm is not None else np.ones_like(hr_np, dtype=bool)

        entry: dict = {"scene": scene}
        entry["psnr"] = _psnr_np(sr_np, hr_np)
        entry["cpsnr"] = _cpsnr_shifted(sr_np, hr_np, mask_np)
        entry["rmse"] = float(np.sqrt(((sr_np[mask_np] - hr_np[mask_np]) ** 2).mean()))
        entry["sam"] = _sam_np(sr_np, hr_np)
        entry["ergas"] = _ergas_np(sr_np, hr_np, scale=scale)
        if _has_skimage:
            entry["ssim"] = float(ssim_fn(sr_np, hr_np, data_range=1.0))

        results.append(entry)

    agg: dict[str, Any] = {"scenes": results}
    for key in ("psnr", "cpsnr", "rmse", "ssim", "sam", "ergas"):
        vals = [r[key] for r in results if key in r and not np.isnan(r[key])]
        if vals:
            agg[f"mean_{key}"] = float(np.mean(vals))
            print(f"[sdp] Proba-V {key}: {agg[f'mean_{key}']:.4f}")

    (run_dir / "metrics.json").write_text(json.dumps(agg, indent=2))
    return agg


# ---------------------------------------------------------------------------
# Public: RAMS runner
# ---------------------------------------------------------------------------

def run_probav_rams(handle: "DatasetHandle", *, run_dir: Path, **_: Any) -> None:
    """
    Run the RAMS super-resolution model (from SDP-2026-probav/) over every scene.

    Outputs per scene:
        <run_dir>/<scene>/SR.png
    Outputs aggregate:
        <run_dir>/summary.json  (per-scene cPSNR + mean, when ground truth exists)
    """
    from . import config as _config

    probav_dir = str(_config.PROBAV_DIR)
    if probav_dir not in sys.path:
        sys.path.insert(0, probav_dir)

    try:
        from models.rams.model import RAMSModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import RAMSModel from SDP-2026-probav/models/rams/model.py. "
            "Make sure TensorFlow is installed and SDP-2026-probav/ is present."
        ) from exc

    ds = _get_dataset(handle)
    band = handle.extras.get("band", "NIR")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = RAMSModel(band=band)

    results: list[dict] = []
    n = len(ds)
    for i, sample in enumerate(ds):
        scene = sample["scene"]
        lrs = sample["lrs"]   # (T, 1, H, W) float32
        qms = sample["qms"]   # (T, 1, H, W) float32

        # RAMS expects (T, H, W) numpy float32
        lrs_np = lrs[:, 0, :, :].numpy()   # (T, H, W)
        qms_np = qms[:, 0, :, :].numpy().astype(bool)

        sr_np = model.predict(lrs_np, qms_np)  # (H*3, W*3) float32 [0,1]
        sr_tensor = torch.from_numpy(sr_np)[None]  # (1, H*3, W*3)

        scene_dir = run_dir / scene
        scene_dir.mkdir(exist_ok=True)
        _save_sr_png(sr_tensor, scene_dir / "SR.png")

        entry: dict = {"scene": scene}
        if "hr" in sample:
            hr_np = sample["hr"].squeeze().numpy().astype(np.float64)
            mask_np = (
                sample["sm"].squeeze().numpy() > 0
                if "sm" in sample
                else np.ones_like(hr_np, dtype=bool)
            )
            entry["cpsnr"] = _cpsnr_shifted(sr_np.astype(np.float64), hr_np, mask_np)
        results.append(entry)

        sys.stderr.write(
            f"\r[sdp] rams {i + 1}/{n}: {scene}"
            + (f"  cPSNR={entry['cpsnr']:.3f} dB" if "cpsnr" in entry else "")
        )
        sys.stderr.flush()

    sys.stderr.write("\n")

    summary: dict[str, Any] = {"model": "rams", "scenes": results}
    cpsnr_vals = [r["cpsnr"] for r in results if "cpsnr" in r and not np.isnan(r["cpsnr"])]
    if cpsnr_vals:
        summary["mean_cpsnr"] = float(np.mean(cpsnr_vals))
        sys.stderr.write(f"[sdp] mean cPSNR = {summary['mean_cpsnr']:.4f} dB\n")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.stderr.write(f"[sdp] RAMS results saved to {run_dir}\n")


# ---------------------------------------------------------------------------
# Public: visualization
# ---------------------------------------------------------------------------

def visualize_probav(
    handle: "DatasetHandle",
    *,
    run_dir: Path,
    num: int = 5,
    out: Optional[Path] = None,
) -> Path:
    """
    Render panels: best LR | SR prediction | HR target | |SR - HR| diff.
    Saves PNGs to out/ (defaults to <run_dir>/vis/).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = _get_dataset(handle)
    run_dir = Path(run_dir)
    out_dir = Path(out) if out else run_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(num, len(ds))
    for i in range(n):
        sample = ds[i]
        scene = sample["scene"]
        lrs = sample["lrs"]   # (T, 1, H, W)
        qms = sample["qms"]

        # Pick clearest LR frame by total clear pixels
        best_idx = int(qms.sum(dim=(-1, -2, -3)).argmax())
        best_lr = lrs[best_idx, 0].numpy()  # (H, W)

        sr_path = run_dir / scene / "SR.png"
        has_sr = sr_path.exists()
        has_hr = "hr" in sample

        # Build panel list dynamically
        panels: list[tuple[str, np.ndarray, str, dict]] = [
            (f"Best LR\n{scene}", best_lr, "gray", {"vmin": 0, "vmax": 1}),
        ]
        if has_sr:
            sr_arr = _load_sr_png(sr_path).squeeze().numpy()
            panels.append(("SR prediction", sr_arr, "gray", {"vmin": 0, "vmax": 1}))
        if has_hr:
            hr_arr = sample["hr"].squeeze().numpy()
            panels.append(("HR target", hr_arr, "gray", {"vmin": 0, "vmax": 1}))
            if has_sr:
                panels.append(
                    ("|SR - HR|", np.abs(sr_arr - hr_arr), "hot", {"vmin": 0, "vmax": 0.1})
                )

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, data, cmap, kwargs) in zip(axes, panels):
            ax.imshow(data, cmap=cmap, **kwargs)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(out_dir / f"{scene}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        sys.stderr.write(f"\r[sdp] visualize_probav {i + 1}/{n}: {scene}")
        sys.stderr.flush()

    sys.stderr.write("\n")
    sys.stderr.write(f"[sdp] visualization saved to {out_dir}\n")
    return out_dir
