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