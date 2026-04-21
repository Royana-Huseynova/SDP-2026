"""
metrics.py
──────────
Evaluation metrics for super-resolution on Proba-V dataset.
All metrics take model OUTPUT (SR) and ground truth (HR) as inputs.

Metrics implemented:
  - PSNR      : Peak Signal-to-Noise Ratio
  - SSIM      : Structural Similarity Index
  - cPSNR     : clearance-corrected PSNR (official Proba-V competition metric)
  - SAM       : Spectral Angle Mapper (adapted for single band)
  - ERGAS     : Erreur Relative Globale Adimensionnelle de Synthèse

Metrics NOT implemented (documented in README):
  - QNR  : requires multiple spectral bands — not applicable to single-band SR
  - FID  : designed for GANs, requires Inception net trained on RGB ImageNet

Usage:
    from evaluation.metrics import evaluate

    scores = evaluate(sr, hr, hr_mask, lr_mean, scale=3)
    print(scores)
    # {'psnr': 34.2, 'ssim': 0.91, 'cpsnr': 35.1, 'sam': 0.02, 'ergas': 3.4}
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# ─────────────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────────────

def psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Args:
        sr:         model output (H, W) float in [0, 1]
        hr:         ground truth (H, W) float in [0, 1]
        data_range: maximum possible value (default 1.0)

    Returns:
        PSNR in dB. Higher is better.
    """
    sr = np.asarray(sr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)

    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')

    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index Measure.

    Args:
        sr:         model output (H, W) float in [0, 1]
        hr:         ground truth (H, W) float in [0, 1]
        data_range: maximum possible value (default 1.0)

    Returns:
        SSIM in [-1, 1]. Higher is better, 1 = perfect.
    """
    sr = np.asarray(sr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)

    return float(ssim_fn(sr, hr, data_range=data_range))


def cpsnr(sr: np.ndarray,
          hr: np.ndarray,
          hr_mask: np.ndarray,
          border: int = 3) -> float:
    """
    Clearance-corrected PSNR — the official Proba-V competition metric.

    Accounts for:
      1. Only valid (unmasked) HR pixels
      2. Brightness bias correction between SR and HR
      3. Sub-pixel registration uncertainty via border cropping

    Args:
        sr:      model output (H, W) float in [0, 1]
        hr:      ground truth (H, W) float in [0, 1]
        hr_mask: HR quality mask (H, W) bool — True = valid pixel
        border:  number of border pixels to crop for registration tolerance

    Returns:
        cPSNR in dB. Higher is better.
    """
    sr      = np.asarray(sr,      dtype=np.float64)
    hr      = np.asarray(hr,      dtype=np.float64)
    hr_mask = np.asarray(hr_mask, dtype=np.float64)

    H, W = sr.shape
    best_cpsnr = -np.inf

    # Try all (2*border+1)^2 shifts to account for registration uncertainty
    for di in range(2 * border + 1):
        for dj in range(2 * border + 1):
            # Crop SR to match shifted HR region
            sr_crop   = sr[border:H - border, border:W - border]
            hr_crop   = hr[di:di + (H - 2 * border), dj:dj + (W - 2 * border)]
            mask_crop = hr_mask[di:di + (H - 2 * border), dj:dj + (W - 2 * border)]

            n_valid = mask_crop.sum()
            if n_valid == 0:
                continue

            # Apply mask
            sr_masked = sr_crop * mask_crop
            hr_masked = hr_crop * mask_crop

            # Brightness bias correction
            bias = (hr_masked.sum() - sr_masked.sum()) / n_valid
            sr_corrected = (sr_masked + bias) * mask_crop

            # cMSE over valid pixels only
            cmse = np.sum((hr_masked - sr_corrected) ** 2) / n_valid

            if cmse == 0:
                return float('inf')

            # cPSNR in 16-bit scale (65535^2 as per competition)
            c = float(10.0 * np.log10((65535.0 ** 2) / (cmse * 65535.0 ** 2)))
            best_cpsnr = max(best_cpsnr, c)

    return best_cpsnr


def sam(sr: np.ndarray, hr: np.ndarray) -> float:
    """
    Spectral Angle Mapper — adapted for single-band images.

    For single-band SR, SAM reduces to measuring the relative brightness
    difference as an angle. A value of 0 means perfect match.

    Args:
        sr: model output (H, W) float in [0, 1]
        hr: ground truth (H, W) float in [0, 1]

    Returns:
        SAM in radians. Lower is better, 0 = perfect.
    """
    sr = np.asarray(sr, dtype=np.float64).ravel()
    hr = np.asarray(hr, dtype=np.float64).ravel()

    dot    = np.dot(sr, hr)
    norm_s = np.linalg.norm(sr)
    norm_h = np.linalg.norm(hr)

    if norm_s == 0 or norm_h == 0:
        return 0.0

    cos_angle = np.clip(dot / (norm_s * norm_h), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def ergas(sr: np.ndarray, hr: np.ndarray, scale: int = 3) -> float:
    """
    Erreur Relative Globale Adimensionnelle de Synthèse.
    Measures normalized global error — commonly used for satellite SR.

    Args:
        sr:    model output (H, W) float in [0, 1]
        hr:    ground truth (H, W) float in [0, 1]
        scale: upscaling factor (default 3 for Proba-V)

    Returns:
        ERGAS. Lower is better, 0 = perfect.
    """
    sr = np.asarray(sr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)

    hr_mean = hr.mean()
    if hr_mean == 0:
        return 0.0

    rmse  = np.sqrt(np.mean((sr - hr) ** 2))
    return float(100.0 * (1.0 / scale) * (rmse / hr_mean))


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(sr: np.ndarray,
             hr: np.ndarray,
             hr_mask: np.ndarray = None,
             scale: int = 3,
             border: int = 3) -> dict:
    """
    Run all metrics on a single SR output vs HR ground truth.
    Call this AFTER the model has produced its output.

    Args:
        sr:      model output (H, W) float in [0, 1]
        hr:      ground truth (H, W) float in [0, 1]
        hr_mask: HR quality mask (H, W) bool — if None, all pixels assumed valid
        scale:   SR upscaling factor (3 for Proba-V)
        border:  border for cPSNR registration tolerance

    Returns:
        dict with keys: psnr, ssim, cpsnr, sam, ergas
    """
    sr = np.asarray(sr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)

    if hr_mask is None:
        hr_mask = np.ones_like(hr, dtype=bool)

    return {
        'psnr' : psnr(sr, hr),
        'ssim' : ssim(sr, hr),
        'cpsnr': cpsnr(sr, hr, hr_mask, border=border),
        'sam'  : sam(sr, hr),
        'ergas': ergas(sr, hr, scale=scale),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_scores(scores: dict, scene_name: str = '') -> None:
    """Print metric scores in a readable format."""
    header = f"  Metrics — {scene_name}" if scene_name else "  Metrics"
    print(f"\n{'─'*45}")
    print(header)
    print(f"{'─'*45}")
    print(f"  PSNR   : {scores['psnr']:>8.4f} dB   (higher is better)")
    print(f"  SSIM   : {scores['ssim']:>8.4f}      (higher is better)")
    print(f"  cPSNR  : {scores['cpsnr']:>8.4f} dB   (higher is better)")
    print(f"  SAM    : {scores['sam']:>8.4f} rad  (lower  is better)")
    print(f"  ERGAS  : {scores['ergas']:>8.4f}      (lower  is better)")
    print(f"{'─'*45}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from data.probav import ProbaVDataset
    from evaluation.aggregate import baseline_upscale
    from data.io import highres_image
    import os

    data_path = '/Users/royana/Desktop/probav_data/train'
    channel   = 'RED'

    channel_path = os.path.join(data_path, channel)
    scenes = sorted([
        os.path.join(channel_path, f)
        for f in os.listdir(channel_path)
        if not f.startswith('.') and os.path.isdir(os.path.join(channel_path, f))
    ])

    print("=" * 50)
    print("Metrics — Quick Test on first 3 scenes")
    print("=" * 50)

    all_scores = []
    for scene_path in scenes[:3]:
        scene_name = os.path.basename(scene_path)

        # Load HR
        hr, hr_mask = highres_image(scene_path, img_as_float=True)

        # Run baseline model
        sr = baseline_upscale(scene_path)

        # Evaluate metrics AFTER model output
        scores = evaluate(sr, hr, hr_mask)
        print_scores(scores, scene_name)
        all_scores.append(scores)

    # Average across scenes
    print("=" * 50)
    print("  Average across 3 scenes")
    print("=" * 50)
    avg = {k: np.mean([s[k] for s in all_scores]) for k in all_scores[0]}
    print_scores(avg, 'Average')