"""
visualize/visualize.py — Unified result visualization for the satellite DL benchmark.

Supports both tasks:
  sr             – super-resolution   (Proba-V)   single-band grayscale
  cloud_removal  – cloud removal      (AllClear)  multi-band RGB composite

Entry point
-----------
    from visualize import visualize_results

    visualize_results(
        dataset_name = "allclear",
        lr           = sample_dict,        # dict for AllClear, np array for ProbaV
        predictions  = pred_np,            # (C,H,W) or (H,W)
        target       = target_np,          # (C,H,W) or (H,W)
        hr_mask      = hr_mask_np,         # (H,W) bool
        scores       = {"mae": ..., ...},
        name         = "roi38068_...",
        save_dir     = "results/vis",      # None → display interactively
        show         = True,
    )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Shared image utilities
# ─────────────────────────────────────────────────────────────────────────────

# S2 RGB band indices (0-based within the 13-band array): R=B4, G=B3, B=B2
_S2_RGB = (3, 2, 1)


def _percentile_stretch(arr, lo=2, hi=98):
    """Stretch a float array to [0, 1] using percentile clipping."""
    vlo = np.nanpercentile(arr, lo)
    vhi = np.nanpercentile(arr, hi)
    if vhi <= vlo:
        return np.zeros_like(arr)
    return np.clip((arr - vlo) / (vhi - vlo), 0, 1)


def _to_rgb(image_chw, rgb_bands=_S2_RGB):
    """
    Convert a (C, H, W) multi-band float array to a (H, W, 3) display image.
    Uses percentile stretching per channel for best contrast.
    Falls back to grayscale if fewer bands than requested.
    """
    image_chw = np.asarray(image_chw, dtype=np.float64)
    C = image_chw.shape[0]
    channels = [min(b, C - 1) for b in rgb_bands]
    rgb = np.stack([_percentile_stretch(image_chw[c]) for c in channels], axis=2)
    return rgb.astype(np.float32)


def _to_gray(image_hw):
    """Stretch a (H, W) single-band float array to a [0, 1] display image."""
    return _percentile_stretch(np.asarray(image_hw, dtype=np.float64)).astype(np.float32)


def _show_mask(ax, mask_hw, title=""):
    """Display a binary (H, W) mask: white = valid/clear, black = cloudy/invalid."""
    ax.imshow(mask_hw.astype(np.float32), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _show_rgb(ax, image, title=""):
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _scores_text(scores):
    lines = ["Metrics"]
    for k, v in scores.items():
        lines.append(f"{k.upper()}: {v:.4f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AllClear (cloud removal) visualization
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_cloud_removal(lr, predictions, target, hr_mask, scores, name,
                              save_dir, show):
    """
    Layout (2 rows):
      Row 0: [Input t=0 RGB] … [Input t=T-1 RGB] | [Prediction RGB] | [Target RGB]
      Row 1: [Cloud mask t=0] … [t=T-1]          | [Valid mask]     | [Score text]
    """
    # lr is the sample_dict produced by AllClearDataset
    input_images = lr["input_images"].numpy()    # (C, T, H, W)
    cld_shdw     = lr["input_cld_shdw"].numpy()  # (2, T, H, W)
    T = input_images.shape[1]

    n_cols = T + 2   # T input frames + prediction + target
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    fig.suptitle(f"{name}", fontsize=10, y=1.01)

    # ── row 0: images ─────────────────────────────────────────────────────────
    for t in range(T):
        _show_rgb(axes[0, t],
                  _to_rgb(input_images[:, t]),
                  title=f"Input t={t}")

    _show_rgb(axes[0, T],     _to_rgb(predictions), title="Prediction")
    _show_rgb(axes[0, T + 1], _to_rgb(target),      title="Target (GT)")

    # ── row 1: masks + scores ─────────────────────────────────────────────────
    for t in range(T):
        cloud_valid = ((cld_shdw[0, t] + cld_shdw[1, t]) <= 0)  # True = clear
        _show_mask(axes[1, t], cloud_valid, title=f"Clear mask t={t}")

    _show_mask(axes[1, T], hr_mask, title="Target valid mask")

    # scores text box
    ax_sc = axes[1, T + 1]
    ax_sc.axis("off")
    ax_sc.text(0.05, 0.95, _scores_text(scores),
               transform=ax_sc.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    _save_or_show(fig, name, save_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Proba-V (super-resolution) visualization
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_sr(lr, predictions, target, hr_mask, scores, name,
                  save_dir, show):
    """
    Layout (2 rows):
      Row 0: [LR t=0] … [LR t=T-1] | [SR Prediction] | [HR Ground Truth]
      Row 1: [QM t=0] … [QM t=T-1] | [HR valid mask] | [Score text]
    """
    # lr is (T, H, W) numpy array for ProbaV
    lr_np = np.asarray(lr)
    T     = lr_np.shape[0]

    from skimage.transform import rescale

    n_cols = T + 2
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    fig.suptitle(f"{name}", fontsize=10, y=1.01)

    # ── row 0: LR frames (upscaled for display) + SR + HR ────────────────────
    for t in range(T):
        lr_up = rescale(lr_np[t], scale=3, order=1, anti_aliasing=False)
        _show_rgb(axes[0, t], _to_gray(lr_up)[:, :, np.newaxis].repeat(3, axis=2),
                  title=f"LR t={t}")

    _show_rgb(axes[0, T],
              _to_gray(predictions)[:, :, np.newaxis].repeat(3, axis=2),
              title="SR Prediction")
    _show_rgb(axes[0, T + 1],
              _to_gray(target)[:, :, np.newaxis].repeat(3, axis=2),
              title="HR Ground Truth")

    # ── row 1: quality masks + scores ────────────────────────────────────────
    # lr_mask is ignored (passed as _ in pipeline); show blank placeholders
    for t in range(T):
        axes[1, t].axis("off")
        axes[1, t].set_title(f"QM t={t}", fontsize=8)

    _show_mask(axes[1, T], hr_mask, title="HR valid mask")

    ax_sc = axes[1, T + 1]
    ax_sc.axis("off")
    ax_sc.text(0.05, 0.95, _scores_text(scores),
               transform=ax_sc.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    _save_or_show(fig, name, save_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Save / show helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig, name, save_dir, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace("/", "_").replace("\\", "_")
        fpath = os.path.join(save_dir, f"{safe_name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"    Saved → {fpath}")
        plt.close(fig)
    if show:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def visualize_results(dataset_name, lr, predictions, target, hr_mask, scores,
                      name, save_dir=None, show=True):
    """
    Unified visualization dispatch for all supported datasets.

    Args:
        dataset_name: "probav" or "allclear"
        lr:           raw input returned by dataset.load_sample()
                        AllClear → sample_dict (dict of tensors)
                        ProbaV   → (T, H, W) numpy array
        predictions:  model output — (C, H, W) for AllClear, (H, W) for ProbaV
        target:       ground-truth — same shape as predictions
        hr_mask:      (H, W) bool — True = valid/clear pixel
        scores:       dict of metric scores from evaluate / evaluate_allclear
        name:         scene / sample identifier string
        save_dir:     directory to save PNG files; None = don't save
        show:         whether to call plt.show() (set False in headless environments)
    """
    from api.pipeline import DATASET_TASK
    task = DATASET_TASK[dataset_name]

    if task == "cloud_removal":
        _visualize_cloud_removal(lr, predictions, target, hr_mask, scores,
                                 name, save_dir, show)
    else:
        _visualize_sr(lr, predictions, target, hr_mask, scores,
                      name, save_dir, show)
