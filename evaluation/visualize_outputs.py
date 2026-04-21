"""
Visualizes model outputs side by side with HR ground truth and metric scores.
Call this AFTER the model has produced its SR output.

Layout per scene:
  LR best frame | SR output | HR ground truth | Metrics panel

Usage:
    from benchmark.visualize_outputs import visualize_output

    visualize_output(
        lr_stack   = lr_stack,    # (T, H, W) float [0,1]
        sr         = sr,          # (HR_H, HR_W) float [0,1] — model output
        hr         = hr,          # (HR_H, HR_W) float [0,1]
        hr_mask    = hr_mask,     # (HR_H, HR_W) bool
        scores     = scores,      # dict from metrics.evaluate()
        scene_name = 'imgset0000',
        model_name = 'Bicubic Baseline',
        save_path  = './outputs/imgset0000.png',
    )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_output(lr_stack: np.ndarray,
                     sr: np.ndarray,
                     hr: np.ndarray,
                     hr_mask: np.ndarray,
                     scores: dict,
                     scene_name: str = '',
                     model_name: str = 'Model',
                     save_path: str = None,
                     show: bool = True) -> None:
    """
    Visualize SR output vs HR ground truth with metric scores.

    Args:
        lr_stack:   (T, H, W) float [0,1] — LR input frames
        sr:         (HR_H, HR_W) float [0,1] — model SR output
        hr:         (HR_H, HR_W) float [0,1] — HR ground truth
        hr_mask:    (HR_H, HR_W) bool — HR quality mask
        scores:     dict from metrics.evaluate()
        scene_name: scene identifier for title
        model_name: name of the model used
        save_path:  if provided, save figure to this path
        show:       if True, open matplotlib window
    """
    # ── Colors ────────────────────────────────────────────────────────────────
    bg      = '#0f1117'
    white   = '#e8e6df'
    muted   = '#9a9890'
    blue    = '#5b8dee'
    green   = '#4caf7d'
    orange  = '#e8a838'
    red_col = '#e05c5c'

    # ── Pick best LR frame by mask coverage ───────────────────────────────────
    best_lr = lr_stack[0]   # default to first frame

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6), facecolor=bg)
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                            left=0.02, right=0.98,
                            top=0.88, bottom=0.04,
                            wspace=0.06)

    fig.suptitle(
        f"Scene: {scene_name}   |   Model: {model_name}",
        fontsize=13, color=white, y=0.97
    )

    # ── Panel 1: Best LR frame ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(best_lr, cmap='gray', interpolation='nearest')
    ax0.set_title(f'LR input\n({best_lr.shape[0]}×{best_lr.shape[1]})',
                  color=white, fontsize=10, pad=6)
    ax0.axis('off')
    for sp in ax0.spines.values():
        sp.set_edgecolor(muted)
        sp.set_linewidth(0.8)

    # ── Panel 2: SR output ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(sr, cmap='gray', interpolation='nearest')
    ax1.set_title(f'SR output\n({sr.shape[0]}×{sr.shape[1]})',
                  color=blue, fontsize=10, pad=6)
    ax1.axis('off')
    for sp in ax1.spines.values():
        sp.set_edgecolor(blue)
        sp.set_linewidth(1.5)

    # ── Panel 3: HR ground truth ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(hr, cmap='gray', interpolation='nearest')
    ax2.set_title(f'HR ground truth\n({hr.shape[0]}×{hr.shape[1]})',
                  color=green, fontsize=10, pad=6)
    ax2.axis('off')
    for sp in ax2.spines.values():
        sp.set_edgecolor(green)
        sp.set_linewidth(1.5)

    # ── Panel 4: Metrics ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.set_facecolor('#181c24')
    ax3.set_xticks([])
    ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_edgecolor('#2a2e3a')

    ax3.text(0.5, 0.96, 'Metrics', transform=ax3.transAxes,
             fontsize=11, fontweight='bold', color=white,
             ha='center', va='top')

    # Metric rows
    metric_rows = [
        ('PSNR',  f"{scores.get('psnr',  0):.3f} dB",  '↑'),
        ('SSIM',  f"{scores.get('ssim',  0):.4f}",      '↑'),
        ('cPSNR', f"{scores.get('cpsnr', 0):.3f} dB",  '↑'),
        ('SAM',   f"{scores.get('sam',   0):.4f} rad",  '↓'),
        ('ERGAS', f"{scores.get('ergas', 0):.4f}",      '↓'),
    ]

    y = 0.80
    for name, val, direction in metric_rows:
        color = green if direction == '↑' else orange
        ax3.text(0.08, y, name, transform=ax3.transAxes,
                 fontsize=9, color=muted, va='top')
        ax3.text(0.92, y, val, transform=ax3.transAxes,
                 fontsize=9, color=white, va='top', ha='right')
        ax3.text(0.08, y - 0.06, direction + ' better',
                 transform=ax3.transAxes,
                 fontsize=7, color=color, va='top')
        y -= 0.16

    # HR mask coverage
    if hr_mask is not None:
        coverage = hr_mask.mean() * 100
        ax3.text(0.08, y - 0.04, f'HR coverage: {coverage:.1f}%',
                 transform=ax3.transAxes,
                 fontsize=8, color=muted, va='top')

    # ── Save / show ───────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)