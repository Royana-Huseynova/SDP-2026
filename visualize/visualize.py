"""
visualize/visualize.py — Unified result visualization for the satellite DL benchmark.

Paper-quality white-background figures modelled after the AllClear publication style.
Metrics appear above the prediction panel; column headers are bold; no dark chrome.

Entry point
-----------
    from visualize import visualize_results

    visualize_results(
        dataset_name = "allclear",
        lr           = sample_dict,
        predictions  = pred_np,
        target       = target_np,
        hr_mask      = hr_mask_np,
        scores       = {"mae": ..., ...},
        name         = "roi38068_...",
        model_name   = "leastcloudy",
        save_dir     = "results",
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

# S2 RGB band indices (0-based): R=B4, G=B3, B=B2
_S2_RGB = (3, 2, 1)

# ── Paper color palette ───────────────────────────────────────────────────────
_FG          = '#1a1a1a'   # primary text
_GREY        = '#555555'   # secondary text / input title
_BLUE        = '#2563eb'   # model A / prediction column
_PURPLE      = '#7c3aed'   # model B column
_GREEN       = '#16a34a'   # target column
_BORDER      = '#cccccc'   # image frame
_METRIC_BG   = '#f0f6ff'   # light blue tint — model A metrics
_METRIC_EDGE = '#93b4e0'
_METRIC_BG_B  = '#faf0ff'  # light purple tint — model B metrics
_METRIC_EDGE_B = '#c4a0f0'


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
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
    return _percentile_stretch(
        np.asarray(image_hw, dtype=np.float64)
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Shared rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _paper_image(ax, img, title, title_color=_FG, subtitle=None):
    """Render one image panel in paper style."""
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(_BORDER)
        sp.set_linewidth(0.8)
    ax.set_facecolor('white')

    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=9.5, fontweight='bold',
                 color=title_color, pad=5, loc='center',
                 fontfamily='DejaVu Sans')


def _metrics_block(ax, scores, task, bg=None, edge=None):
    """Render a compact metrics text block into a blank axis."""
    ax.axis('off')
    ax.set_facecolor('white')
    bg   = bg   or _METRIC_BG
    edge = edge or _METRIC_EDGE

    if task == 'cloud_removal':
        line1_items = [
            ('MAE',   'mae',   ''),
            ('RMSE',  'rmse',  ''),
            ('PSNR',  'psnr',  ' dB'),
        ]
        line2_items = [
            ('SAM',   'sam',   '°'),
            ('SSIM',  'ssim',  ''),
            ('LPIPS', 'lpips', ''),
        ]
    else:
        line1_items = [
            ('PSNR',  'psnr',  ' dB'),
            ('SSIM',  'ssim',  ''),
            ('cPSNR', 'cpsnr', ' dB'),
        ]
        line2_items = [
            ('SAM',   'sam',   ' rad'),
            ('ERGAS', 'ergas', ''),
        ]

    def _fmt(items):
        return '   ·   '.join(
            f"{lbl} {scores[k]:.4f}{unit}"
            for lbl, k, unit in items
            if k in scores
        )

    text = _fmt(line1_items) + '\n' + _fmt(line2_items)

    ax.text(0.5, 0.5, text,
            transform=ax.transAxes,
            fontsize=8.5, ha='center', va='center',
            color=_FG, linespacing=1.7,
            fontfamily='DejaVu Sans',
            bbox=dict(
                boxstyle='round,pad=0.45',
                facecolor=bg,
                edgecolor=edge,
                linewidth=0.9,
            ))


# ─────────────────────────────────────────────────────────────────────────────
# AllClear (cloud removal)
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_cloud_removal(lr, predictions, target, hr_mask, scores, name,
                              model_name, save_dir, show):
    """
    Layout:
      Row 0 (metrics):   —blank—  |  metrics block  |  —blank—
      Row 1 (images):    Input    |  Prediction      |  Target GT
    """
    input_images = lr["input_images"].numpy()    # (C, T, H, W)
    cld_shdw     = lr["input_cld_shdw"].numpy()  # (2, T, H, W)

    cloud_per_t = (cld_shdw[0] + cld_shdw[1]).mean(axis=(1, 2))
    cloudiest_t = int(np.argmax(cloud_per_t))
    T           = input_images.shape[1]
    coverage    = hr_mask.mean() * 100

    fig = plt.figure(figsize=(15, 6.5), facecolor='white')

    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[0.22, 1],
        hspace=0.10,
        wspace=0.06,
        top=0.88, bottom=0.05, left=0.02, right=0.98,
        figure=fig,
    )

    fig.suptitle(
        f"Scene: {name}   |   Model: {model_name}",
        fontsize=11, fontweight='bold', color=_FG, y=0.97,
        fontfamily='DejaVu Sans',
    )

    # ── row 0: metrics centred above prediction ───────────────────────────────
    for col in [0, 2]:
        fig.add_subplot(gs[0, col]).axis('off')

    _metrics_block(fig.add_subplot(gs[0, 1]), scores, task='cloud_removal')

    # ── row 1: images ─────────────────────────────────────────────────────────
    _paper_image(
        fig.add_subplot(gs[1, 0]),
        _to_rgb(input_images[:, cloudiest_t]),
        title=f'Input  (t={cloudiest_t}, cloudiest of {T})',
        title_color=_GREY,
    )
    _paper_image(
        fig.add_subplot(gs[1, 1]),
        _to_rgb(predictions),
        title=f'Prediction  ({model_name})',
        title_color=_BLUE,
    )
    _paper_image(
        fig.add_subplot(gs[1, 2]),
        _to_rgb(target),
        title='Target (GT)',
        title_color=_GREEN,
        subtitle=f'Clear coverage: {coverage:.1f}%',
    )

    _save_or_show(fig, name, save_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Proba-V (super-resolution)
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_sr(lr, predictions, target, hr_mask, scores, name,
                  model_name, save_dir, show):
    """
    Layout:
      Row 0 (metrics):   —blank—            |  metrics block      |  —blank—
      Row 1 (images):    LR (mean of T)     |  SR (model_name)    |  HR ground truth
    """
    from skimage.transform import rescale

    lr_np    = np.asarray(lr)   # (T, H, W)
    T        = lr_np.shape[0]
    coverage = hr_mask.mean() * 100

    lr_up    = rescale(lr_np.mean(axis=0), scale=3, order=1, anti_aliasing=False)
    lr_gray3 = _to_gray(lr_up)[:, :, np.newaxis].repeat(3, axis=2)
    sr_gray3 = _to_gray(predictions)[:, :, np.newaxis].repeat(3, axis=2)
    hr_gray3 = _to_gray(target)[:, :, np.newaxis].repeat(3, axis=2)

    fig = plt.figure(figsize=(15, 6.5), facecolor='white')

    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[0.22, 1],
        hspace=0.10,
        wspace=0.06,
        top=0.88, bottom=0.05, left=0.02, right=0.98,
        figure=fig,
    )

    fig.suptitle(
        f"Scene: {name}   |   Model: {model_name}",
        fontsize=11, fontweight='bold', color=_FG, y=0.97,
        fontfamily='DejaVu Sans',
    )

    for col in [0, 2]:
        fig.add_subplot(gs[0, col]).axis('off')

    _metrics_block(fig.add_subplot(gs[0, 1]), scores, task='sr')

    _paper_image(
        fig.add_subplot(gs[1, 0]),
        lr_gray3,
        title=f'LR  (mean of {T} frames)',
        title_color=_GREY,
    )
    _paper_image(
        fig.add_subplot(gs[1, 1]),
        sr_gray3,
        title=f'SR  ({model_name})',
        title_color=_BLUE,
    )
    _paper_image(
        fig.add_subplot(gs[1, 2]),
        hr_gray3,
        title='HR Ground Truth',
        title_color=_GREEN,
        subtitle=f'Valid coverage: {coverage:.1f}%',
    )

    _save_or_show(fig, name, save_dir, show)


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison visualization
# ─────────────────────────────────────────────────────────────────────────────

def _comparison_figure(input_img, pred_a, pred_b, target,
                       hr_mask, scores_a, scores_b,
                       name, label_a, label_b, task,
                       save_dir, show):
    """
    4-column layout:
      Row 0 (metrics):  blank | Model A metrics (blue) | Model B metrics (purple) | blank
      Row 1 (images):   Input | Model A prediction     | Model B prediction       | Target GT
    """
    coverage = hr_mask.mean() * 100

    fig = plt.figure(figsize=(20, 6.5), facecolor='white')
    gs  = gridspec.GridSpec(
        2, 4,
        height_ratios=[0.25, 1],
        hspace=0.10, wspace=0.06,
        top=0.88, bottom=0.05, left=0.02, right=0.98,
        figure=fig,
    )

    fig.suptitle(
        f"Model Comparison:  {label_a}  vs  {label_b}   |   Scene: {name}",
        fontsize=11, fontweight='bold', color=_FG, y=0.97,
        fontfamily='DejaVu Sans',
    )

    # ── row 0: metrics ────────────────────────────────────────────────────────
    fig.add_subplot(gs[0, 0]).axis('off')
    fig.add_subplot(gs[0, 3]).axis('off')

    ax_ma = fig.add_subplot(gs[0, 1])
    ax_mb = fig.add_subplot(gs[0, 2])

    _metrics_block(ax_ma, scores_a, task,
                   bg=_METRIC_BG,   edge=_METRIC_EDGE)
    _metrics_block(ax_mb, scores_b, task,
                   bg=_METRIC_BG_B, edge=_METRIC_EDGE_B)

    # ── row 1: images ─────────────────────────────────────────────────────────
    _paper_image(fig.add_subplot(gs[1, 0]), input_img,
                 title='Input (cloudiest)' if task == 'cloud_removal'
                       else 'LR (mean of frames)',
                 title_color=_GREY)

    _paper_image(fig.add_subplot(gs[1, 1]), pred_a,
                 title=label_a, title_color=_BLUE)

    _paper_image(fig.add_subplot(gs[1, 2]), pred_b,
                 title=label_b, title_color=_PURPLE)

    _paper_image(fig.add_subplot(gs[1, 3]), target,
                 title='Target (GT)',
                 title_color=_GREEN,
                 subtitle=f'Clear coverage: {coverage:.1f}%')

    _save_or_show(fig, f"compare_{name}", save_dir, show)


def visualize_comparison(dataset_name, lr, pred_a, pred_b, target,
                         hr_mask, scores_a, scores_b,
                         name, label_a, label_b,
                         save_dir=None, show=True):
    """
    Side-by-side comparison of two model outputs.

    Args:
        dataset_name:  "probav" or "allclear"
        lr:            raw input from dataset.load_sample()
        pred_a:        (C,H,W) or (H,W) output of model A
        pred_b:        (C,H,W) or (H,W) output of model B
        target:        ground-truth array
        hr_mask:       (H,W) bool valid-pixel mask
        scores_a/b:    metric dicts from evaluate / evaluate_allclear
        name:          scene identifier
        label_a/b:     display names for the two models
        save_dir:      directory to save PNG; None = skip
        show:          call plt.show()
    """
    from api.pipeline import DATASET_TASK
    task = DATASET_TASK[dataset_name]

    if task == 'cloud_removal':
        input_images = lr["input_images"].numpy()
        cld_shdw     = lr["input_cld_shdw"].numpy()
        cloud_per_t  = (cld_shdw[0] + cld_shdw[1]).mean(axis=(1, 2))
        t            = int(np.argmax(cloud_per_t))
        input_img    = _to_rgb(input_images[:, t])
        pred_a_disp  = _to_rgb(pred_a)
        pred_b_disp  = _to_rgb(pred_b)
        target_disp  = _to_rgb(target)
    else:
        from skimage.transform import rescale
        lr_np = np.asarray(lr)
        lr_up = rescale(lr_np.mean(axis=0), scale=3, order=1, anti_aliasing=False)
        def _g3(x): return _to_gray(x)[:, :, np.newaxis].repeat(3, axis=2)
        input_img   = _g3(lr_up)
        pred_a_disp = _g3(pred_a)
        pred_b_disp = _g3(pred_b)
        target_disp = _g3(target)

    _comparison_figure(
        input_img, pred_a_disp, pred_b_disp, target_disp,
        hr_mask, scores_a, scores_b,
        name, label_a, label_b, task,
        save_dir, show,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Save / show
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig, name, save_dir, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace("/", "_").replace("\\", "_")
        fpath = os.path.join(save_dir, f"{safe_name}.png")
        fig.savefig(fpath, dpi=200, bbox_inches='tight',
                    facecolor='white')
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
        model_name:   name of the model (shown in title and prediction panel)
        save_dir:     directory to save PNG files; None = don't save
        show:         whether to call plt.show()
    """
    from api.pipeline import DATASET_TASK
    task = DATASET_TASK[dataset_name]

    if task == "cloud_removal":
        _visualize_cloud_removal(lr, predictions, target, hr_mask, scores,
                                 name, model_name, save_dir, show)
    else:
        _visualize_sr(lr, predictions, target, hr_mask, scores,
                      name, model_name, save_dir, show)
