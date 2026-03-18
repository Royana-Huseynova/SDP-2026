#!/usr/bin/env python3
"""
Universal AllClear Dataset Visualizer with Metrics
Author: Nijat Alisoy, Huseyn Sadatkhanov, Pasha Zulfugarli, Royana Huseynova
Works on: Windows, macOS, Linux
"""
import sys
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import argparse
import numpy as np
from PIL import Image
try:
    import rasterio as rio
except Exception as e:
    raise ImportError(
        "rasterio is required by scripts/metrics.py but could not be imported.\n"
        "Install it with `pip install rasterio` (or see rasterio docs for conda installs).\n"
        f"Original error: {e}"
    )
import torch
import csv
import pyiqa
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_metric = pyiqa.create_metric("lpips", device=device)

# ===== Utility functions =====
def stretch(x):
    x = x.astype("float32")
    vals = x[~np.isnan(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype="float32")
    lo, hi = np.percentile(vals, (1, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype="float32")
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0, 1)

def read_rgb(path):
    """Read a TIFF and return normalized RGB array (0-1 float)"""
    with rio.open(path) as ds:
        C = ds.count
        nodata = ds.nodata
        if C >= 4:
            bands = (4, 3, 2)
        elif C >= 3:
            bands = (1, 2, 3)
        else:
            x = ds.read(1).astype("float32")
            mask = (x == nodata) if nodata is not None else (x == 0)
            if (~mask).sum() < 100:
                return None
            x[mask] = np.nan
            rgb = np.stack([stretch(x)]*3, axis=-1)
            return np.nan_to_num(rgb, nan=0.0)

        r, g, b = [ds.read(b).astype("float32") for b in bands]
        mask = (r == nodata) | (g == nodata) | (b == nodata) if nodata else (r == 0) & (g == 0) & (b == 0)
        if (~mask).sum() < 100:
            return None
        r = np.where(mask, np.nan, r)
        g = np.where(mask, np.nan, g)
        b = np.where(mask, np.nan, b)
        if max(r.max(), g.max(), b.max()) > 1.5:
            r, g, b = r/10000.0, g/10000.0, b/10000.0
        rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
        return np.nan_to_num(rgb, nan=0.0)

def compute_psnr(img, reference, max_val=1.0):
    """Compute PSNR between two HxWxC images (0-1 normalized)"""
    img_t = torch.from_numpy(img).float()
    ref_t = torch.from_numpy(reference).float()
    mse = torch.mean((img_t - ref_t) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()

def compute_ssim(img, reference):
    """
    Compute SSIM between two HxWxC images (0-1 normalized).
    We compute SSIM per channel and average (safe for RGB).
    """
    scores = []
    for c in range(img.shape[2]):
        score = ssim(
            reference[:, :, c],
            img[:, :, c],
            data_range=1.0
        )
        scores.append(score)
    return float(np.mean(scores))

def compute_lpips(img, reference):
    """
    LPIPS between two HxWxC images in [0,1].
    Returns a float (lower is better).
    """
    # HWC -> NCHW and move to device
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    ref_t = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        val = lpips_metric(img_t, ref_t)  # returns tensor
    return float(val.item())

def compute_sam(img, reference, eps=1e-8, degrees=True):
    """
    SAM between two HxWxC images (0-1).
    Returns mean angle over pixels.
    """
    a = img.astype(np.float32)
    b = reference.astype(np.float32)

    dot = np.sum(a * b, axis=2)
    na = np.linalg.norm(a, axis=2)
    nb = np.linalg.norm(b, axis=2)

    cos = dot / (na * nb + eps)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos)

    if degrees:
        ang = ang * (180.0 / np.pi)

    return float(np.mean(ang))


def save_comparison_vis(img_rgb, ref_rgb, metrics_dict, out_path, title="AllClear Comparison"):
    """
    Save side-by-side visualization with metrics text underneath.
    img_rgb/ref_rgb: HxWx3 in [0,1]
    metrics_dict: {"PSNR":..., "SSIM":..., ...}
    out_path: Path to save PNG
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])

    ax1.imshow(np.clip(img_rgb, 0, 1))
    ax1.set_title("Image")
    ax1.axis("off")

    ax2.imshow(np.clip(ref_rgb, 0, 1))
    ax2.set_title("Reference")
    ax2.axis("off")

    # Metrics text
    lines = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    metrics_text = "   |   ".join(lines)

    ax_txt.axis("off")
    ax_txt.text(
        0.5, 0.5,
        metrics_text,
        ha="center", va="center",
        fontsize=11
    )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



# ===== Main function =====
def main():
    parser = argparse.ArgumentParser(description="AllClear PSNR evaluation for s2_toa images")
    parser.add_argument("--data", type=str, default="allclear_dataset", help="Path to dataset root folder")
    parser.add_argument("--csv", type=str, default="metric_results.csv", help="Output CSV filename")
    parser.add_argument("--out_vis", type=str, default="vis_results", help="Folder to save comparison visualizations")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"❌ Dataset folder not found: {root}")
        sys.exit(1)

    results = []

    # Loop through each ROI
    for roi in sorted(root.iterdir())[:4]:  # first 2 ROIs only
        if not roi.is_dir():
            continue

        # Collect all s2_toa images in this ROI
        s2_images = []
        for time_folder in roi.iterdir():
            s2_toa = time_folder / "s2_toa"
            if s2_toa.exists():
                s2_images.extend(sorted(s2_toa.glob("*.tif")))

        if not s2_images:
            continue

        # Choose a reference image (the first one for simplicity)
        reference_path = s2_images[0]
        reference_rgb = read_rgb(reference_path)
        if reference_rgb is None:
            continue

        # Compute Metrics
        N = 3
        for img_path in s2_images:
            img_rgb = read_rgb(img_path)
            if img_rgb is None:
                continue
            psnr_value = compute_psnr(img_rgb, reference_rgb)
            ssim_value = compute_ssim(img_rgb, reference_rgb)
            lpips_value = compute_lpips(img_rgb, reference_rgb)
            sam_value = compute_sam(img_rgb, reference_rgb)
            metrics_dict = {
                 "PSNR(dB)": psnr_value,
                "SSIM": ssim_value,
                "LPIPS": lpips_value,
                "SAM(deg)": sam_value
            }
            vis_name = f"{roi.name}_{img_path.stem}_vs_{reference_path.stem}.png"
            save_comparison_vis(
                img_rgb,
                reference_rgb,
                metrics_dict,
                Path(args.out_vis) / roi.name / vis_name,
                title=f"ROI: {roi.name}"
            )
            results.append({
                "ROI": roi.name,
                "Image": img_path.name,
                "Reference": reference_path.name,
                "PSNR_dB": round(psnr_value, 2),
                "SSIM": round(ssim_value, 4),
                "LPIPS": round(lpips_value, 4),
                "SAM_deg": round(sam_value, 4)
            })
            print(f"{img_path.name} vs {reference_path.name} | PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f} | LPIPS: {lpips_value:.4f} | SAM: {sam_value: 4f}")

    # Save results to CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ROI", "Image", "Reference", "PSNR_dB", "SSIM", "LPIPS", "SAM_deg"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n🎉 Done — Metric results saved to {args.csv}")

if __name__ == "__main__":
    main()
