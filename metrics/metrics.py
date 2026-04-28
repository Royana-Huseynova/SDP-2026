"""
──────────
Evaluation metrics for super-resolution on Proba-V dataset.
All metrics take model OUTPUT (SR) and ground truth (HR) as inputs.

Metrics implemented:
  - PSNR      : Peak Signal-to-Noise Ratio
  - SSIM      : Structural Similarity Index
  - cPSNR     : clearance-corrected PSNR (official Proba-V competition metric)
  - SAM       : Spectral Angle Mapper (adapted for single band)
  - ERGAS     : Erreur Relative Globale Adimensionnelle de Synthèse

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

import numpy as np

def cpsnr(sr: np.ndarray,
          hr: np.ndarray,
          hr_mask: np.ndarray,
          scene_path=None,
          norm_table=None,
          border: int = 3) -> float:
    """
    Official-style Proba-V cPSNR (leaderboard compatible).
    Keeps same idea as EscVM but more numerically faithful.
    """

    sr = np.asarray(sr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)
    hr_mask = np.asarray(hr_mask, dtype=bool)

    H, W = sr.shape
    c = border

    # ── 1. crop SR once (same as official)
    sr_crop = sr[c:H - c, c:W - c].ravel()

    best_score = np.inf

    # ── 2. shift search (u,v)
    for u in range(2 * c + 1):
        for v in range(2 * c + 1):

            hr_crop = hr[u:u + (H - 2 * c), v:v + (W - 2 * c)].ravel()
            mask_crop = hr_mask[u:u + (H - 2 * c), v:v + (W - 2 * c)].ravel()

            # only valid pixels
            valid = mask_crop

            if np.sum(valid) == 0:
                continue

            _hr = hr_crop[valid]
            _sr = sr_crop[valid]

            # 3. brightness bias correction (official style)
            b = np.mean(_hr - _sr)

            diff = (_hr - (_sr + b))

            cMSE = np.mean(diff * diff)

            if cMSE <= 0:
                return 1.0  # perfect score

            cPSNR = -10.0 * np.log10(cMSE)

            
            if norm_table is not None and scene_path is not None:
                N = norm_table.loc[scene_path][0]
                score = N / cPSNR
            else:
                score = cPSNR

            best_score = min(best_score, score)

    return float(best_score)

    


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
# AllClear cloud-removal metrics  (multi-band, mask-aware)
# ─────────────────────────────────────────────────────────────────────────────

# Lazy singleton so the LPIPS network is loaded once per process
_lpips_fn = None

def _get_lpips_fn(net: str = 'alex'):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips as _lpips_lib
        except ImportError:
            raise ImportError(
                "lpips package not found. Install it with:  pip install lpips"
            )
        _lpips_fn = _lpips_lib.LPIPS(net=net, verbose=False)
    return _lpips_fn

def mae_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean Absolute Error over valid (non-cloud) pixels across all bands.

    Args:
        pred, target: (C, H, W) or (H, W) float in [0, 1]
        mask:         (H, W) bool — True = valid pixel

    Returns:
        MAE scalar. Lower is better.
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask   = np.asarray(mask,   dtype=bool)
    if mask.sum() == 0:
        return float('nan')
    p = pred[:, mask] if pred.ndim == 3 else pred[mask]
    t = target[:, mask] if target.ndim == 3 else target[mask]
    return float(np.abs(p - t).mean())


def rmse_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    Root Mean Squared Error over valid pixels across all bands.

    Args:
        pred, target: (C, H, W) or (H, W) float in [0, 1]
        mask:         (H, W) bool — True = valid pixel

    Returns:
        RMSE scalar. Lower is better.
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask   = np.asarray(mask,   dtype=bool)
    if mask.sum() == 0:
        return float('nan')
    p = pred[:, mask] if pred.ndim == 3 else pred[mask]
    t = target[:, mask] if target.ndim == 3 else target[mask]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def psnr_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                data_range: float = 1.0) -> float:
    """
    PSNR computed over valid pixels only.

    Args:
        pred, target: (C, H, W) or (H, W) float in [0, 1]
        mask:         (H, W) bool — True = valid pixel
        data_range:   maximum pixel value (default 1.0)

    Returns:
        PSNR in dB. Higher is better.
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask   = np.asarray(mask,   dtype=bool)
    if mask.sum() == 0:
        return float('nan')
    p = pred[:, mask] if pred.ndim == 3 else pred[mask]
    t = target[:, mask] if target.ndim == 3 else target[mask]
    mse = np.mean((p - t) ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10((data_range ** 2) / mse))


def sam_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    Spectral Angle Mapper over valid pixels (multi-band).

    Computes the mean spectral angle (degrees) between pred and target
    across all unmasked spatial positions.

    Args:
        pred, target: (C, H, W) float in [0, 1]  — C >= 1
        mask:         (H, W) bool — True = valid pixel

    Returns:
        Mean SAM in degrees. Lower is better, 0 = perfect.
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask   = np.asarray(mask,   dtype=bool)

    if pred.ndim == 2:
        return float(np.degrees(sam(pred * mask, target * mask)))

    C, H, W = pred.shape
    p = pred.reshape(C, -1).T[mask.ravel()]    # (N, C)
    t = target.reshape(C, -1).T[mask.ravel()]

    if len(p) == 0:
        return float('nan')

    dot    = (p * t).sum(axis=1)
    norm_p = np.linalg.norm(p, axis=1)
    norm_t = np.linalg.norm(t, axis=1)
    denom  = norm_p * norm_t
    valid  = denom > 0
    if not valid.any():
        return float('nan')
    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)).mean())


def ssim_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                data_range: float = 1.0) -> float:
    """
    Mean per-channel SSIM for multi-band images.

    Args:
        pred, target: (C, H, W) or (H, W) float in [0, 1]
        mask:         (H, W) bool  (kept for API consistency)
        data_range:   maximum pixel value (default 1.0)

    Returns:
        Mean SSIM across channels. Higher is better, 1 = perfect.
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if pred.ndim == 2:
        return float(ssim_fn(pred, target, data_range=data_range))

    scores = [
        ssim_fn(pred[c], target[c], data_range=data_range)
        for c in range(pred.shape[0])
    ]
    return float(np.mean(scores))


def lpips_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                 net: str = 'alex') -> float:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) on the RGB composite.

    LPIPS uses a pretrained AlexNet (default) or VGG to measure perceptual
    distance between two images — unlike pixel-level metrics it captures
    texture and structural differences the way humans perceive them.

    Because LPIPS was designed for 3-channel RGB images, this function
    extracts Sentinel-2 bands 3, 2, 1 (R, G, B at 0-based index) to form
    the RGB composite. Invalid pixels (mask==False) are replaced with the
    per-channel mean of valid pixels before scoring, which minimises
    boundary artefacts at cloud edges.

    Args:
        pred, target: (C, H, W) float32 in [0, 1]
        mask:         (H, W) bool — True = valid pixel
        net:          backbone network — 'alex' (default, fast) or 'vgg'

    Returns:
        LPIPS scalar in [0, 1]. Lower is better, 0 = identical.
    """
    import torch

    _S2_RGB = (3, 2, 1)
    pred   = np.asarray(pred,   dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask   = np.asarray(mask,   dtype=bool)

    C = pred.shape[0]
    rgb_idx = [min(b, C - 1) for b in _S2_RGB]

    p_rgb = pred[rgb_idx].copy()    # (3, H, W)
    t_rgb = target[rgb_idx].copy()  # (3, H, W)

    # replace invalid pixels with per-channel mean of valid pixels
    if mask.any() and not mask.all():
        for c in range(3):
            fill_p = float(p_rgb[c][mask].mean())
            fill_t = float(t_rgb[c][mask].mean())
            p_rgb[c][~mask] = fill_p
            t_rgb[c][~mask] = fill_t

    # scale [0, 1] → [-1, 1] (LPIPS convention)
    p_t = torch.from_numpy(p_rgb).unsqueeze(0) * 2.0 - 1.0  # (1, 3, H, W)
    t_t = torch.from_numpy(t_rgb).unsqueeze(0) * 2.0 - 1.0

    loss_fn = _get_lpips_fn(net)
    with torch.no_grad():
        score = loss_fn(p_t, t_t)

    return float(score.item())


def evaluate_allclear(pred: np.ndarray,
                      target: np.ndarray,
                      hr_mask: np.ndarray = None) -> dict:
    """
    Run all cloud-removal metrics on a single model output vs target.

    Args:
        pred:    model output  (C, H, W) float in [0, 1]
        target:  ground truth  (C, H, W) float in [0, 1]
        hr_mask: valid-pixel mask (H, W) bool — True = clear pixel.
                 If None, all pixels are treated as valid.

    Returns:
        dict with keys: mae, rmse, psnr, sam, ssim, lpips
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if hr_mask is None:
        H = pred.shape[-2] if pred.ndim == 3 else pred.shape[0]
        W = pred.shape[-1] if pred.ndim == 3 else pred.shape[1]
        hr_mask = np.ones((H, W), dtype=bool)

    scores = {
        "mae":  mae_masked(pred, target, hr_mask),
        "rmse": rmse_masked(pred, target, hr_mask),
        "psnr": psnr_masked(pred, target, hr_mask),
        "sam":  sam_masked(pred, target, hr_mask),
        "ssim": ssim_masked(pred, target, hr_mask),
    }

    try:
        scores["lpips"] = lpips_masked(
            pred.astype(np.float32), target.astype(np.float32), hr_mask
        )
    except ImportError:
        pass   # lpips not installed — skip silently

    return scores


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
    import os
    import numpy as np

    from datasets.probav.probav import ProbaVDataset
    from models.RAMS.model import RAMSModel
    from datasets.probav.io import highres_image

    # ── CONFIG ────────────────────────────────────────────────
    data_path = '/Users/royana/Desktop/probav_data/train'
    channel   = 'NIR'   # or 'RED'
    num_scenes = 3      # how many scenes to test

    print("=" * 50)
    print("Metrics — RAMS Model Evaluation")
    print("=" * 50)

    # ── Load dataset ──────────────────────────────────────────
    dataset = ProbaVDataset(
        base_path=data_path,
        channel=channel,
        max_t=9
    )

    print(f"Total scenes available: {len(dataset)}")

    # ── Load model ────────────────────────────────────────────
    model = RAMSModel(band=channel)

    all_scores = []

    # ── Loop over scenes ──────────────────────────────────────
    for i in range(min(num_scenes, len(dataset))):
        lr, hr, lr_mask, scene_name = dataset[i]

        print(f"\nProcessing scene: {scene_name}")

        # IMPORTANT: reload HR mask (dataset gives LR masks)
        scene_path = os.path.join(data_path, channel, scene_name)
        hr, hr_mask = highres_image(scene_path)

        # ── Run RAMS model ─────────────────────────────────
        sr = model.predict(lr)

        # ── Evaluate metrics ──────────────────────────────
        scores = evaluate(sr, hr, hr_mask)

        print_scores(scores, scene_name)
        all_scores.append(scores)

    # ── Compute average ───────────────────────────────────────
    if all_scores:
        avg_scores = {
            k: np.mean([s[k] for s in all_scores])
            for k in all_scores[0]
        }

        print("=" * 50)
        print("Average over evaluated scenes")
        print("=" * 50)
        print_scores(avg_scores, "Average")