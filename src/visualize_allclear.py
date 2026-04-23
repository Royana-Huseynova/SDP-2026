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


def stretch_rgb_per_channel(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for c in range(3):
        ch = rgb[..., c]
        mask = np.isfinite(ch) & (np.abs(ch) > 1e-6)
        vals = ch[mask]
        if vals.size == 0:
            continue

        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)
        out[..., c] = (ch - lo) / (hi - lo + 1e-6)

    return np.clip(out, 0, 1)


def get_prediction_filename(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name == "ctgan":
        return "ctgan_predictions.pt"
    elif model_name == "uncrtaints":
        return "uncrtaints_predictions.pt"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
def get_metrics_csv_filename(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name == "ctgan":
        return "ctgan_metrics_per_sample.csv"
    elif model_name == "uncrtaints":
        return "uncrtaints_metrics_per_sample.csv"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_metadata_filename(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name == "ctgan":
        return "ctgan_metadata.csv"
    elif model_name == "uncrtaints":
        return "uncrtaints_metadata.csv"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def read_mask_gray(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    m = read_tif_chw(path)
    if m.ndim == 3:
        m = m[0]
    m = np.nan_to_num(m, nan=0.0)

    mmin, mmax = float(np.min(m)), float(np.max(m))
    if mmax - mmin < 1e-6:
        return np.zeros_like(m, dtype=np.float32)
    return (m - mmin) / (mmax - mmin + 1e-6)


def load_predictions(pred_path: Path) -> torch.Tensor:
    preds = torch.load(pred_path, map_location="cpu")

    if isinstance(preds, dict):
        for k in ["pred", "preds", "prediction", "predictions", "outputs"]:
            if k in preds:
                preds = preds[k]
                break

    if not isinstance(preds, torch.Tensor):
        raise ValueError(f"Unexpected predictions format: {type(preds)}")

    if preds.ndim == 5 and preds.shape[1] == 1:
        preds = preds[:, 0]   # (N,C,H,W)
    elif preds.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W) or (N,1,C,H,W), got {preds.shape}")

    return preds


def extract_raw_s2_rgb(chw: np.ndarray) -> np.ndarray:
    x = to_reflectance01(chw)
    if x.shape[0] >= 4:
        return np.stack([x[3], x[2], x[1]], axis=-1)  # B4,B3,B2
    elif x.shape[0] == 3:
        return np.stack([x[0], x[1], x[2]], axis=-1)
    else:
        return np.stack([x[0], x[0], x[0]], axis=-1)


def rgb_from_raw_s2(chw: np.ndarray, stretch: bool = True) -> np.ndarray:
    rgb = extract_raw_s2_rgb(chw)
    return stretch_rgb_per_channel(rgb) if stretch else np.clip(rgb, 0, 1)


def extract_ctgan_rgb(chw: np.ndarray) -> np.ndarray:
    x = to_reflectance01(chw)
    if x.shape[0] >= 3:
        return np.stack([x[0], x[1], x[2]], axis=-1)
    elif x.shape[0] == 1:
        return np.stack([x[0], x[0], x[0]], axis=-1)
    else:
        raise ValueError(f"Unexpected CTGAN shape: {x.shape}")


def extract_uncrtaints_rgb(chw: np.ndarray) -> np.ndarray:
    x = to_reflectance01(chw)
    if x.shape[0] >= 4:
        return np.stack([x[3], x[2], x[1]], axis=-1)  # B4,B3,B2
    elif x.shape[0] == 3:
        return np.stack([x[0], x[1], x[2]], axis=-1)
    elif x.shape[0] == 1:
        return np.stack([x[0], x[0], x[0]], axis=-1)
    else:
        raise ValueError(f"Unexpected UnCRtainTS shape: {x.shape}")


def rgb_from_uncrtaints_prediction(chw: np.ndarray, stretch: bool = True) -> np.ndarray:
    rgb = extract_uncrtaints_rgb(chw)
    return stretch_rgb_per_channel(rgb) if stretch else np.clip(rgb, 0, 1)


def compute_rgb_scale(rgb_ref: np.ndarray):
    rgb_ref = np.nan_to_num(rgb_ref, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    los, his = [], []
    for c in range(3):
        ch = rgb_ref[..., c]
        mask = np.isfinite(ch) & (np.abs(ch) > 1e-6)
        vals = ch[mask]
        if vals.size == 0:
            los.append(0.0)
            his.append(1.0)
        else:
            los.append(np.percentile(vals, 1))
            his.append(np.percentile(vals, 99))
    return np.array(los, dtype=np.float32), np.array(his, dtype=np.float32)


def apply_rgb_scale(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out = (rgb - lo[None, None, :]) / (hi[None, None, :] - lo[None, None, :] + 1e-6)
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing prediction outputs for the chosen model")
    ap.add_argument("--model", choices=["ctgan", "uncrtaints"], required=True)
    ap.add_argument("--json", required=True, help="Dataset JSON used for this run")
    ap.add_argument("--out", default=None, help="Output folder for PNG panels")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no_stretch", action="store_true", help="Disable RGB stretching")
    args = ap.parse_args()

    stretch = not args.no_stretch
    run_dir = Path(args.run_dir)

    pred_path = run_dir / get_prediction_filename(args.model)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    metrics_csv_path = run_dir / get_metrics_csv_filename(args.model)
    metadata_path = run_dir / get_metadata_filename(args.model)

    if metrics_csv_path.exists():
        dfm = pd.read_csv(metrics_csv_path)
    elif metadata_path.exists():
        dfm = pd.read_csv(metadata_path)
    else:
        dfm = None

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.values())
    preds = load_predictions(pred_path)

    out_dir = Path(args.out) if args.out else (run_dir / f"images_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    N = min(len(items), preds.shape[0])
    end = min(args.start + args.num, N)

    for i in range(args.start, end):
        print(f"\n=== START sample {i} ===")
        it = items[i]

        s2_list = it["s2_toa"]
        input_path = s2_list[-1][1]
        target_path = it["target"][0][1]

        cld_path = target_path.replace("\\s2_toa\\", "\\cld_shdw\\").replace("_s2_toa_", "_cld_shdw_")
        dw_path = target_path.replace("\\s2_toa\\", "\\dw\\").replace("_s2_toa_", "_dw_")

        inp = read_tif_chw(input_path)
        tgt = read_tif_chw(target_path)
        pred = preds[i].detach().cpu().float().numpy()

        print("pred shape:", pred.shape)
        for c in range(min(pred.shape[0], 4)):
            print(
                f"pred[{c}] min={pred[c].min():.4f} "
                f"max={pred[c].max():.4f} "
                f"mean={pred[c].mean():.4f}"
            )

        if args.model == "uncrtaints":
            inp_rgb = rgb_from_raw_s2(inp, stretch=stretch)
            tgt_rgb = rgb_from_raw_s2(tgt, stretch=stretch)
            pred_rgb = rgb_from_uncrtaints_prediction(pred, stretch=stretch)

        elif args.model == "ctgan":
            inp_rgb_raw = extract_raw_s2_rgb(inp)
            tgt_rgb_raw = extract_raw_s2_rgb(tgt)
            pred_rgb_raw = extract_ctgan_rgb(pred)

            if stretch:
                lo, hi = compute_rgb_scale(tgt_rgb_raw)
                inp_rgb = apply_rgb_scale(inp_rgb_raw, lo, hi)
                tgt_rgb = apply_rgb_scale(tgt_rgb_raw, lo, hi)
                pred_rgb = apply_rgb_scale(pred_rgb_raw, lo, hi)
            else:
                inp_rgb = np.clip(inp_rgb_raw, 0, 1)
                tgt_rgb = np.clip(tgt_rgb_raw, 0, 1)
                pred_rgb = np.clip(pred_rgb_raw, 0, 1)

        else:
            raise ValueError(f"Unsupported model: {args.model}")

        diff_rgb = np.clip(np.abs(pred_rgb - tgt_rgb), 0, 1)

        cld = read_mask_gray(cld_path)
        dw = read_mask_gray(dw_path)

        panels = [
            ("Input (S2)", inp_rgb),
            (f"Prediction ({args.model})", pred_rgb),
            ("Target (GT)", tgt_rgb),
            ("|Pred-GT|", diff_rgb),
        ]

        if cld is not None:
            panels.append(("Cloud/Shadow mask", cld))
        if dw is not None:
            panels.append(("DW", dw))

        print(f"Building figure for sample {i}")
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, img) in zip(axes, panels):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

            subtitle = f"Sample {i}"
            if dfm is not None and i < len(dfm):
                r = dfm.iloc[i]
                parts = [f"Sample {i}"]

                def pick(row, *names):
                    for name in names:
                        if name in row.index and pd.notna(row[name]):
                            return row[name]
                    return None

                psnr_v = pick(r, "PSNR", "psnr")
                ssim_v = pick(r, "SSIM", "ssim")
                sam_v = pick(r, "SAM", "sam")
                mae_v = pick(r, "MAE", "mae")
                rmse_v = pick(r, "RMSE", "rmse")
                lpips_v = pick(r, "LPIPS", "lpips")

                if psnr_v is not None:
                    parts.append(f"PSNR {psnr_v:.2f}")
                if ssim_v is not None:
                    parts.append(f"SSIM {ssim_v:.3f}")
                if sam_v is not None:
                    parts.append(f"SAM {sam_v:.2f}")
                if mae_v is not None:
                    parts.append(f"MAE {mae_v:.4f}")
                if rmse_v is not None:
                    parts.append(f"RMSE {rmse_v:.4f}")
                if lpips_v is not None:
                    parts.append(f"LPIPS {lpips_v:.3f}")

                subtitle = " | ".join(parts)

        fig.suptitle(subtitle, fontsize=11)

        roi_name = it["roi"][0] if isinstance(it["roi"], list) else str(it["roi"])
        out_path = out_dir / f"{roi_name}_{i:04d}.png"

        fig.tight_layout()
        print(f"About to save sample {i} -> {out_path}")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"=== DONE sample {i} ===")
        print("Saved:", out_path)


if __name__ == "__main__":
    main()