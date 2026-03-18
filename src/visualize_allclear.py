import argparse
from pathlib import Path
import json
import os
import numpy as np
import torch
import rasterio as rio
import matplotlib.pyplot as plt
import pandas as pd

def read_tif_chw(path: str) -> np.ndarray:
    with rio.open(path) as ds:
        x = ds.read().astype(np.float32)  # (C,H,W)
        nodata = ds.nodata
    if nodata is not None:
        x[x == nodata] = np.nan
    return x

def to_reflectance01(x: np.ndarray) -> np.ndarray:
    m = np.nanmax(x)
    if m > 1.5:
        x = x / 10000.0
    return np.clip(x, 0.0, 1.0)

def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    vals = rgb.reshape(-1, 3)
    mask = np.isfinite(vals).all(axis=1) & (np.abs(vals).sum(axis=1) > 1e-6)
    vals = vals[mask]
    if vals.size == 0:
        return np.zeros_like(rgb, dtype=np.float32)
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)
    out = (rgb - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0, 1)

def chw_to_rgb(chw: np.ndarray) -> np.ndarray:
    x = to_reflectance01(chw)
    if x.shape[0] >= 4:
        r, g, b = x[3], x[2], x[1]  # S2 RGB = B4,B3,B2
    elif x.shape[0] == 3:
        r, g, b = x[0], x[1], x[2]
    else:
        r = g = b = x[0]
    rgb = np.stack([r, g, b], axis=-1)
    return stretch_rgb(rgb)

def read_mask_gray(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return None
    m = read_tif_chw(path)
    if m.ndim == 3:
        m = m[0]
    m = np.nan_to_num(m, nan=0.0)
    # normalize to 0..1 for display
    mmin, mmax = float(np.min(m)), float(np.max(m))
    if mmax - mmin < 1e-6:
        return np.zeros_like(m, dtype=np.float32)
    return (m - mmin) / (mmax - mmin + 1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing uncrtaints_predictions.pt")
    ap.add_argument("--json", required=True, help="Dataset JSON used for this run (paths)")
    ap.add_argument("--out", default=None, help="Output folder for PNG panels")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / "uncrtaints_predictions.pt"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    
    metrics_path = run_dir / "uncrtaints_metadata.csv"
    dfm = pd.read_csv(metrics_path) if metrics_path.exists() else None


    data = json.load(open(args.json, "r", encoding="utf-8"))
    items = list(data.values())  # order assumed consistent with benchmark

    preds = torch.load(pred_path, map_location="cpu")
    if isinstance(preds, dict):
        for k in ["pred", "preds", "prediction", "predictions", "outputs"]:
            if k in preds:
                preds = preds[k]
                break

    if not isinstance(preds, torch.Tensor):
        raise ValueError(f"Unexpected predictions format: {type(preds)}")
    # Accept either (N,C,H,W) or (N,1,C,H,W)
    if preds.ndim == 5 and preds.shape[1] == 1:
        preds = preds[:, 0]   # -> (N,C,H,W)
    elif preds.ndim != 4:
        raise ValueError(f"Expected predictions tensor (N,C,H,W) or (N,1,C,H,W). Got {preds.shape}")

    out_dir = Path(args.out) if args.out else (run_dir / "images")
    out_dir.mkdir(parents=True, exist_ok=True)

    N = min(len(items), preds.shape[0])
    end = min(args.start + args.num, N)

    for i in range(args.start, end):
        it = items[i]

        # choose “cloudy input” as the last s2_toa frame (you can change to [0] if you want)
        s2_list = it["s2_toa"]
        input_path = s2_list[-1][1]
        target_path = it["target"][0][1]

        # derive mask paths (matches your folder naming)
        cld_path = target_path.replace("\\s2_toa\\", "\\cld_shdw\\").replace("_s2_toa_", "_cld_shdw_")
        dw_path  = target_path.replace("\\s2_toa\\", "\\dw\\").replace("_s2_toa_", "_dw_")

        inp = read_tif_chw(input_path)
        tgt = read_tif_chw(target_path)
        pred = preds[i].detach().cpu().float().numpy()  # (C,H,W)

        inp_rgb = chw_to_rgb(inp)
        tgt_rgb = chw_to_rgb(tgt)
        pred_rgb = chw_to_rgb(pred)
        diff_rgb = stretch_rgb(np.abs(pred_rgb - tgt_rgb))

        cld = read_mask_gray(cld_path)
        dw  = read_mask_gray(dw_path)

        panels = [
            ("Input (S2)", inp_rgb),
            ("Prediction", pred_rgb),
            ("Target (GT)", tgt_rgb),
            ("|Pred-GT|", diff_rgb),
        ]

        if cld is not None:
            panels.append(("Cloud/Shadow mask", cld))
        if dw is not None:
            panels.append(("DW", dw))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, img) in zip(axes, panels):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        subtitle = f"Sample {i}"
        if dfm is not None and i < len(dfm):
            r = dfm.iloc[i]
            # safe get (in case a column is missing)
            psnr = r["psnr"] if "psnr" in dfm.columns else None
            ssim = r["ssim"] if "ssim" in dfm.columns else None
            sam  = r["sam"]  if "sam"  in dfm.columns else None
            mae  = r["mae"]  if "mae"  in dfm.columns else None
            rmse = r["rmse"] if "rmse" in dfm.columns else None

            parts = [f"Sample {i}"]
            if psnr is not None: parts.append(f"PSNR {psnr:.2f}")
            if ssim is not None: parts.append(f"SSIM {ssim:.3f}")
            if sam  is not None: parts.append(f"SAM {sam:.2f}")
            if mae  is not None: parts.append(f"MAE {mae:.4f}")
            if rmse is not None: parts.append(f"RMSE {rmse:.4f}")
            subtitle = " | ".join(parts)

        fig.suptitle(subtitle, fontsize=11)

        out_path = out_dir / f"sample_{i:04d}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()