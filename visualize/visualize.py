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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Color palette ─────────────────────────────────────────────────────────────
_BG     = '#0f1117'
_PANEL  = '#181c24'
_BORDER = '#2a2e3a'
_WHITE  = '#e8e6df'
_MUTED  = '#9a9890'
_BLUE   = '#5b8dee'
_GREEN  = '#4caf7d'
_ORANGE = '#e8a838'

# S2 RGB band indices (0-based within the 13-band array): R=B4, G=B3, B=B2
_S2_RGB = (3, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Shared image utilities
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_stretch(arr, lo=2, hi=98):
    vlo = np.nanpercentile(arr, lo)
    vhi = np.nanpercentile(arr, hi)
    if vhi <= vlo:
        return np.zeros_like(arr)
    return np.clip((arr - vlo) / (vhi - vlo), 0, 1)


def _to_rgb(image_chw, rgb_bands=_S2_RGB):
    image_chw = np.asarray(image_chw, dtype=np.float64)
    C = image_chw.shape[0]
    channels = [min(b, C - 1) for b in rgb_bands]
    rgb = np.stack([_percentile_stretch(image_chw[c]) for c in channels], axis=2)
    return rgb.astype(np.float32)


def _to_gray(image_hw):
    return _percentile_stretch(np.asarray(image_hw, dtype=np.float64)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Axis styling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, border_color=_MUTED, border_width=0.8):
    ax.set_facecolor(_BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(border_color)
        sp.set_linewidth(border_width)


def _show_image(ax, img, title, title_color=_WHITE,
                border_color=_MUTED, border_width=0.8):
    ax.imshow(img, interpolation='nearest')
    ax.set_title(title, color=title_color, fontsize=9, pad=5)
    _style_ax(ax, border_color, border_width)


def _show_mask(ax, mask_hw, title):
    ax.imshow(mask_hw.astype(np.float32), cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, color=_MUTED, fontsize=8, pad=4)
    _style_ax(ax, _BORDER, 0.6)


def _metrics_panel(ax, scores, task):
    """Render a styled metrics panel into ax."""
    ax.set_facecolor(_PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
        sp.set_linewidth(0.8)

    ax.text(0.5, 0.96, 'Metrics', transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=_WHITE,
            ha='center', va='top')

    if task == 'cloud_removal':
        rows = [
            ('MAE',  'mae',  '↓', _ORANGE),
            ('RMSE', 'rmse', '↓', _ORANGE),
            ('PSNR', 'psnr', '↑', _GREEN),
            ('SAM',  'sam',  '↓', _ORANGE),
            ('SSIM', 'ssim', '↑', _GREEN),
        ]
    else:
        rows = [
            ('PSNR',  'psnr',  '↑', _GREEN),
            ('SSIM',  'ssim',  '↑', _GREEN),
            ('cPSNR', 'cpsnr', '↑', _GREEN),
            ('SAM',   'sam',   '↓', _ORANGE),
            ('ERGAS', 'ergas', '↓', _ORANGE),
        ]

    y = 0.82
    for label, key, arrow, color in rows:
        val = scores.get(key)
        val_str = f"{val:.4f}" if val is not None else '—'
        ax.text(0.10, y, label, transform=ax.transAxes,
                fontsize=8, color=_MUTED, va='top')
        ax.text(0.92, y, val_str, transform=ax.transAxes,
                fontsize=8, color=_WHITE, va='top', ha='right')
        ax.text(0.10, y - 0.055, f'{arrow} better',
                transform=ax.transAxes,
                fontsize=6.5, color=color, va='top')
        y -= 0.155

    # HR coverage line
    if hasattr(ax, '_hr_mask') and ax._hr_mask is not None:
        cov = ax._hr_mask.mean() * 100
        ax.text(0.10, y - 0.02, f'Coverage: {cov:.1f}%',
                transform=ax.transAxes,
                fontsize=7.5, color=_MUTED, va='top')


# ─────────────────────────────────────────────────────────────────────────────
# AllClear (cloud removal) visualization
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_cloud_removal(lr, predictions, target, hr_mask, scores, name,
                              model_name, save_dir, show):
    """
    Layout (1 row, 4 columns):
      [Input (cloudiest frame)] | [Prediction] | [Target GT] | [Metrics panel]
    """
    input_images = lr["input_images"].numpy()    # (C, T, H, W)
    cld_shdw     = lr["input_cld_shdw"].numpy()  # (2, T, H, W)

    # Pick the cloudiest input frame as the representative input
    cloud_per_t = (cld_shdw[0] + cld_shdw[1]).mean(axis=(1, 2))  # (T,)
    cloudiest_t = int(np.argmax(cloud_per_t))

    fig, axes = plt.subplots(
        1, 4,
        figsize=(18, 6),
        facecolor=_BG,
        gridspec_kw={"wspace": 0.06},
    )
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.02, right=0.98)
    fig.suptitle(f"Scene: {name}   |   Model: {model_name}",
                 fontsize=11, color=_WHITE, y=0.98)

    _show_image(axes[0], _to_rgb(input_images[:, cloudiest_t]),
                title=f'Input (cloudiest, t={cloudiest_t})', title_color=_WHITE)
    _show_image(axes[1], _to_rgb(predictions),
                title=f'Prediction ({model_name})', title_color=_BLUE,
                border_color=_BLUE, border_width=1.5)
    _show_image(axes[2], _to_rgb(target),
                title='Target (GT)', title_color=_GREEN,
                border_color=_GREEN, border_width=1.5)

    ax_m = axes[3]
    _metrics_panel(ax_m, scores, task='cloud_removal')
    coverage = hr_mask.mean() * 100
    ax_m.text(0.10, 0.10, f'Coverage: {coverage:.1f}%',
              transform=ax_m.transAxes,
              fontsize=7.5, color=_MUTED, va='top')

    _save_or_show(fig, name, save_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Proba-V (super-resolution) visualization
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_sr(lr, predictions, target, hr_mask, scores, name,
                  model_name, save_dir, show):
    """
    Layout (1 row, 4 columns):
      [LR (mean of frames)] | [SR Prediction] | [HR Ground Truth] | [Metrics panel]
    """
    from skimage.transform import rescale

    lr_np = np.asarray(lr)   # (T, H, W)

    # Mean of all LR frames as the representative input
    lr_mean = lr_np.mean(axis=0)
    lr_up   = rescale(lr_mean, scale=3, order=1, anti_aliasing=False)
    lr_gray3 = _to_gray(lr_up)[:, :, np.newaxis].repeat(3, axis=2)
    sr_gray3 = _to_gray(predictions)[:, :, np.newaxis].repeat(3, axis=2)
    hr_gray3 = _to_gray(target)[:, :, np.newaxis].repeat(3, axis=2)

    fig, axes = plt.subplots(
        1, 4,
        figsize=(18, 6),
        facecolor=_BG,
        gridspec_kw={"wspace": 0.06},
    )
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.02, right=0.98)
    fig.suptitle(f"Scene: {name}   |   Model: {model_name}",
                 fontsize=11, color=_WHITE, y=0.98)

    _show_image(axes[0], lr_gray3,
                title='LR (mean of frames)', title_color=_WHITE)
    _show_image(axes[1], sr_gray3,
                title=f'SR ({model_name})', title_color=_BLUE,
                border_color=_BLUE, border_width=1.5)
    _show_image(axes[2], hr_gray3,
                title='HR Ground Truth', title_color=_GREEN,
                border_color=_GREEN, border_width=1.5)

    ax_m = axes[3]
    _metrics_panel(ax_m, scores, task='sr')
    coverage = hr_mask.mean() * 100
    ax_m.text(0.10, 0.10, f'Coverage: {coverage:.1f}%',
              transform=ax_m.transAxes,
              fontsize=7.5, color=_MUTED, va='top')

    _save_or_show(fig, name, save_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Save / show helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig, name, save_dir, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace("/", "_").replace("\\", "_")
        fpath = os.path.join(save_dir, f"{safe_name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {fpath}")
        plt.close(fig)
    if show:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def visualize_results(dataset_name, lr, predictions, target, hr_mask, scores,
                      name, model_name="Model", save_dir=None, show=True):
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
        model_name:   name of the model used (shown in title and prediction panel)
        save_dir:     directory to save PNG files; None = don't save
        show:         whether to call plt.show() (set False in headless environments)
    """
    from api.pipeline import DATASET_TASK
    task = DATASET_TASK[dataset_name]

    if task == "cloud_removal":
        _visualize_cloud_removal(lr, predictions, target, hr_mask, scores,
                                 name, model_name, save_dir, show)
    else:
        _visualize_sr(lr, predictions, target, hr_mask, scores,
                      name, model_name, save_dir, show)
