#!/usr/bin/env python3
"""
Unified metrics evaluator driven by:
- prediction .pt file
- dataset json
- ground-truth tif files from json
- masks reconstructed from target paths

No need to modify benchmark.py.

Supports:
- ctgan
- uncrtaints

Metrics:
- MAE
- RMSE
- PSNR
- SAM
- SSIM
- LPIPS
- FID_RGB
- optional QNR

Notes:
- LPIPS/FID are computed on RGB projections.
- QNR is optional and not a standard cloud-removal metric.
"""

import argparse
from pathlib import Path
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import rasterio as rio

try:
    import pyiqa
except Exception:
    pyiqa = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None


# ============================================================
# IO
# ============================================================

def read_tif_chw(path: str) -> np.ndarray:
    with rio.open(path) as ds:
        x = ds.read().astype(np.float32)  # (C,H,W)
        nodata = ds.nodata
    if nodata is not None:
        x[x == nodata] = np.nan
    return x


def to_reflectance01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = np.nanmax(x)
    if m > 1.5:
        x = x / 10000.0
    return np.clip(x, 0.0, 1.0)


def load_predictions(pred_path: Path) -> torch.Tensor:
    preds = torch.load(pred_path, map_location="cpu")

    if isinstance(preds, dict):
        for k in ["pred", "preds", "prediction", "predictions", "outputs", "output"]:
            if k in preds:
                preds = preds[k]
                break

    if not isinstance(preds, torch.Tensor):
        raise ValueError(f"Unexpected predictions format: {type(preds)}")

    # normalize to (N,C,H,W)
    if preds.ndim == 5 and preds.shape[1] == 1:
        preds = preds[:, 0]
    elif preds.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected (N,C,H,W) or (N,1,C,H,W), got {preds.shape}")

    return preds.float()


def get_prediction_filename(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name == "ctgan":
        return "ctgan_predictions.pt"
    elif model_name == "uncrtaints":
        return "uncrtaints_predictions.pt"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ============================================================
# Band handling
# ============================================================

def extract_raw_s2_bands(chw: np.ndarray) -> np.ndarray:
    """
    GT/input Sentinel-2 from TIFF.
    Assumes natural raw order where RGB = B4,B3,B2 -> indices 3,2,1
    """
    x = to_reflectance01(chw)
    return x


def extract_prediction_bands(chw: np.ndarray, model: str) -> np.ndarray:
    """
    Normalize prediction band interpretation to comparable multispectral tensor.
    For metric computation we compare aligned channels, not stretched RGB.
    """
    x = to_reflectance01(chw)

    if model == "ctgan":
        # CTGAN wrapper you showed uses bands tuple and often outputs channels already arranged
        # for visualization as first 3 = RGB-like. For full metrics we keep raw prediction tensor.
        return x
    elif model == "uncrtaints":
        return x
    else:
        raise ValueError(f"Unsupported model: {model}")


def get_rgb_from_gt(chw: np.ndarray) -> np.ndarray:
    x = to_reflectance01(chw)
    if x.shape[0] >= 4:
        return np.stack([x[3], x[2], x[1]], axis=0)  # (3,H,W)
    elif x.shape[0] >= 3:
        return np.stack([x[0], x[1], x[2]], axis=0)
    elif x.shape[0] == 1:
        return np.repeat(x, 3, axis=0)
    else:
        raise ValueError(f"Unexpected GT shape: {x.shape}")


def get_rgb_from_prediction(chw: np.ndarray, model: str) -> np.ndarray:
    x = to_reflectance01(chw)
    if model == "ctgan":
        if x.shape[0] >= 3:
            return np.stack([x[0], x[1], x[2]], axis=0)
        elif x.shape[0] == 1:
            return np.repeat(x, 3, axis=0)
    elif model == "uncrtaints":
        if x.shape[0] >= 4:
            return np.stack([x[3], x[2], x[1]], axis=0)
        elif x.shape[0] >= 3:
            return np.stack([x[0], x[1], x[2]], axis=0)
        elif x.shape[0] == 1:
            return np.repeat(x, 3, axis=0)
    raise ValueError(f"Unexpected prediction shape for {model}: {x.shape}")


# ============================================================
# Masks
# ============================================================

def derive_cld_shdw_path_from_target(target_path: str) -> str:
    return target_path.replace("\\s2_toa\\", "\\cld_shdw\\").replace("_s2_toa_", "_cld_shdw_")


def derive_dw_path_from_target(target_path: str) -> str:
    return target_path.replace("\\s2_toa\\", "\\dw\\").replace("_s2_toa_", "_dw_")


def read_cld_shdw_mask(path: str, h: int, w: int) -> np.ndarray:
    """
    Returns valid-data mask for evaluation:
      1 = valid non-cloud/non-shadow
      0 = cloud/shadow
    Expected cld_shdw tif usually has 2 channels:
      [cloud, shadow]
    """
    if not os.path.exists(path):
        return np.ones((1, h, w), dtype=np.float32)

    m = read_tif_chw(path)  # (C,H,W)
    m = np.nan_to_num(m, nan=0.0)

    if m.ndim != 3:
        return np.ones((1, h, w), dtype=np.float32)

    if m.shape[0] >= 2:
        cld = m[0] > 0
        shd = m[1] > 0
        bad = np.logical_or(cld, shd)
    else:
        bad = m[0] > 0

    valid = np.logical_not(bad).astype(np.float32)
    return valid[None, ...]  # (1,H,W)


# ============================================================
# Metrics
# ============================================================

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dims):
    denom = mask.sum(dim=dims).clamp_min(1e-8)
    return (x * mask).sum(dim=dims) / denom


def mae(pred, target, mask):
    return masked_mean(torch.abs(pred - target), mask, dims=(1, 2, 3))


def rmse(pred, target, mask):
    mse = masked_mean((pred - target) ** 2, mask, dims=(1, 2, 3))
    return torch.sqrt(mse.clamp_min(1e-12))


def psnr(pred, target, mask, max_val=1.0):
    mse = masked_mean((pred - target) ** 2, mask, dims=(1, 2, 3)).clamp_min(1e-12)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def sam(pred, target, mask):
    pred_n = F.normalize(pred, p=2, dim=1)
    tar_n = F.normalize(target, p=2, dim=1)
    dot = (pred_n * tar_n).sum(dim=1).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.acos(dot)).unsqueeze(1)

    spatial_mask = (mask[:, :1] > 0).float()
    denom = spatial_mask.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    return (ang * spatial_mask).sum(dim=(1, 2, 3)) / denom


def ssim(pred, target, mask, window_size=11, k1=0.01, k2=0.03):
    c = pred.shape[1]
    L = 1.0
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    def gaussian_window(size, sigma, device):
        coords = torch.arange(size, dtype=torch.float32, device=device)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.view(1, 1, -1) * g.view(1, -1, 1)

    window = gaussian_window(window_size, 1.5, pred.device).repeat(c, 1, 1, 1)
    bin_mask = (mask > 0.5).float()

    mu_x = F.conv2d(pred * bin_mask, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(target * bin_mask, window, padding=window_size // 2, groups=c)

    sigma_x = F.conv2d(pred * pred * bin_mask, window, padding=window_size // 2, groups=c) - mu_x ** 2
    sigma_y = F.conv2d(target * target * bin_mask, window, padding=window_size // 2, groups=c) - mu_y ** 2
    sigma_xy = F.conv2d(pred * target * bin_mask, window, padding=window_size // 2, groups=c) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / den.clamp_min(1e-12)

    return masked_mean(ssim_map, bin_mask, dims=(1, 2, 3))


def compute_lpips_batch(pred_rgb, target_rgb, device):
    if pyiqa is None:
        return [np.nan] * pred_rgb.shape[0]

    lpips_metric = pyiqa.create_metric("lpips", device=device)
    vals = []
    with torch.no_grad():
        for i in range(pred_rgb.shape[0]):
            a = pred_rgb[i:i+1].to(device)
            b = target_rgb[i:i+1].to(device)
            vals.append(float(lpips_metric(a, b).item()))
    return vals


def compute_fid(pred_rgb, target_rgb, device="cpu", batch_size=16):
    if FrechetInceptionDistance is None:
        return np.nan

    pred_u8 = (pred_rgb.clamp(0, 1) * 255.0).round().to(torch.uint8)
    target_u8 = (target_rgb.clamp(0, 1) * 255.0).round().to(torch.uint8)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    for start in range(0, pred_u8.shape[0], batch_size):
        end = min(start + batch_size, pred_u8.shape[0])
        fid.update(target_u8[start:end].to(device), real=True)
        fid.update(pred_u8[start:end].to(device), real=False)

    return float(fid.compute().item())


def _safe_corrcoef(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() < 2 or b.numel() < 2:
        return torch.tensor(0.0)
    if torch.std(a) < 1e-8 or torch.std(b) < 1e-8:
        return torch.tensor(0.0)
    return torch.corrcoef(torch.stack([a, b]))[0, 1]


def compute_qnr(pred, ref):
    """
    Very rough optional surrogate.
    Included only because you asked for QNR too.
    Not a standard cloud-removal metric.
    """
    if pred.shape != ref.shape or pred.shape[1] < 2:
        return np.nan

    vals = []
    n, c, _, _ = pred.shape
    for b in range(n):
        dist_sum = 0.0
        count = 0
        for i in range(c):
            for j in range(i + 1, c):
                cp = _safe_corrcoef(pred[b, i], pred[b, j])
                ct = _safe_corrcoef(ref[b, i], ref[b, j])
                dist_sum += torch.abs(cp - ct)
                count += 1
        if count == 0:
            vals.append(torch.tensor(float("nan")))
        else:
            d_lambda = dist_sum / count
            qnr_like = (1.0 - d_lambda).clamp(min=0.0)
            vals.append(qnr_like)
    vals = torch.stack(vals)
    return float(torch.nanmean(vals).item())


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing prediction outputs")
    ap.add_argument("--model", choices=["ctgan", "uncrtaints"], required=True)
    ap.add_argument("--json", required=True, help="Dataset JSON used for this run")
    ap.add_argument("--out_prefix", default=None, help="Prefix for output files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / get_prediction_filename(args.model)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.values())
    preds = load_predictions(pred_path)

    N = min(len(items), preds.shape[0])

    rows = []
    pred_rgb_all = []
    tgt_rgb_all = []

    device = torch.device(args.device)

    for i in range(N):
        it = items[i]
        target_path = it["target"][0][1]
        cld_path = derive_cld_shdw_path_from_target(target_path)

        tgt_np = extract_raw_s2_bands(read_tif_chw(target_path))
        pred_np = extract_prediction_bands(preds[i].detach().cpu().numpy(), args.model)

        # align channel count safely
        c = min(pred_np.shape[0], tgt_np.shape[0])
        pred_np = pred_np[:c]
        tgt_np = tgt_np[:c]

        h, w = tgt_np.shape[1], tgt_np.shape[2]
        valid_mask_np = read_cld_shdw_mask(cld_path, h, w)  # (1,H,W)
        valid_mask_np = np.repeat(valid_mask_np, c, axis=0)

        pred_t = torch.from_numpy(pred_np).unsqueeze(0).float().to(device)   # (1,C,H,W)
        tgt_t = torch.from_numpy(tgt_np).unsqueeze(0).float().to(device)
        mask_t = torch.from_numpy(valid_mask_np).unsqueeze(0).float().to(device)

        mae_v = float(mae(pred_t, tgt_t, mask_t).item())
        rmse_v = float(rmse(pred_t, tgt_t, mask_t).item())
        psnr_v = float(psnr(pred_t, tgt_t, mask_t).item())
        sam_v = float(sam(pred_t, tgt_t, mask_t).item())
        ssim_v = float(ssim(pred_t, tgt_t, mask_t).item())

        pred_rgb = torch.from_numpy(get_rgb_from_prediction(pred_np, args.model)).unsqueeze(0).float()
        tgt_rgb = torch.from_numpy(get_rgb_from_gt(tgt_np)).unsqueeze(0).float()

        pred_rgb_all.append(pred_rgb)
        tgt_rgb_all.append(tgt_rgb)

        roi_name = it["roi"][0] if isinstance(it["roi"], list) else str(it["roi"])
        rows.append({
            "index": i,
            "roi": roi_name,
            "target_path": target_path,
            "mask_path": cld_path,
            "MAE": mae_v,
            "RMSE": rmse_v,
            "PSNR": psnr_v,
            "SAM": sam_v,
            "SSIM": ssim_v,
        })

        print(
            f"[{i+1}/{N}] {roi_name} | "
            f"PSNR={psnr_v:.3f} SSIM={ssim_v:.4f} SAM={sam_v:.3f} "
            f"MAE={mae_v:.5f} RMSE={rmse_v:.5f}"
        )

    per_sample_df = pd.DataFrame(rows)

    pred_rgb_all = torch.cat(pred_rgb_all, dim=0)
    tgt_rgb_all = torch.cat(tgt_rgb_all, dim=0)

    lpips_vals = compute_lpips_batch(pred_rgb_all, tgt_rgb_all, device=device)
    per_sample_df["LPIPS"] = lpips_vals

    fid_rgb = compute_fid(pred_rgb_all, tgt_rgb_all, device=device)
    qnr_like = compute_qnr(
        torch.from_numpy(np.stack([
            extract_prediction_bands(preds[i].detach().cpu().numpy(), args.model)[:min(
                extract_prediction_bands(preds[i].detach().cpu().numpy(), args.model).shape[0],
                extract_raw_s2_bands(read_tif_chw(items[i]["target"][0][1])).shape[0]
            )]
            for i in range(N)
        ], axis=0)).float(),
        torch.from_numpy(np.stack([
            extract_raw_s2_bands(read_tif_chw(items[i]["target"][0][1]))[:min(
                extract_prediction_bands(preds[i].detach().cpu().numpy(), args.model).shape[0],
                extract_raw_s2_bands(read_tif_chw(items[i]["target"][0][1])).shape[0]
            )]
            for i in range(N)
        ], axis=0)).float()
    )

    agg = {
        "MAE": float(per_sample_df["MAE"].mean()),
        "RMSE": float(per_sample_df["RMSE"].mean()),
        "PSNR": float(per_sample_df["PSNR"].mean()),
        "SAM": float(per_sample_df["SAM"].mean()),
        "SSIM": float(per_sample_df["SSIM"].mean()),
        "LPIPS": float(per_sample_df["LPIPS"].mean()) if "LPIPS" in per_sample_df else np.nan,
        "FID_RGB": fid_rgb,
        "QNR_like": qnr_like,
    }

    prefix = args.out_prefix if args.out_prefix else args.model

    per_sample_path = run_dir / f"{prefix}_metrics_per_sample.csv"
    agg_path = run_dir / f"{prefix}_unified_aggregated.csv"
    notes_path = run_dir / f"{prefix}_metrics_notes.json"

    per_sample_df.to_csv(per_sample_path, index=False)
    pd.DataFrame([agg]).to_csv(agg_path, index=False)

    notes = {
        "model": args.model,
        "prediction_file": str(pred_path),
        "dataset_json": args.json,
        "mask_rule": "Mask derived from target path by replacing s2_toa -> cld_shdw and _s2_toa_ -> _cld_shdw_",
        "lpips_note": "Computed on RGB projection only",
        "fid_note": "Computed on RGB projection only",
        "qnr_note": "QNR_like is only a rough surrogate, not a standard cloud-removal metric",
    }
    with open(notes_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    print("\n=== Aggregated Metrics ===")
    for k, v in agg.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(per_sample_path)
    print(agg_path)
    print(notes_path)


if __name__ == "__main__":
    main()