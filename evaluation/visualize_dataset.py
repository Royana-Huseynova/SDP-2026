"""
Visualize raw Proba-V dataset BEFORE any model is involved.

What it shows per scene:
  - All LR frames (with their quality masks overlaid)
  - HR ground truth
  - HR quality mask
  - Per-scene statistics (brightness, coverage, frame count)

Usage:
  python visualize_dataset.py --data_path /path/to/probav_data/train \
                               --channel RED \
                               --scene_idx 0 \
                               --max_frames 9 \
                               --save_dir ./dataset_previews
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Try importing your local dataset/io modules ──────────────────────────────
# Adjust the import path to match your project structure.
# If running from the project root: `python -m benchmark.visualize_dataset`
try:
    from data.probav import ProbaVDataset
    from data.io import highres_image, lowres_image_iterator
except ImportError:
    # Fallback: running as a standalone script
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from data.probav import ProbaVDataset
        from data.io import highres_image, lowres_image_iterator
    except ImportError:
        raise ImportError(
            "Could not import dataset/io modules. "
            "Run from your project root or adjust the import paths at the top of this file."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Scale a float32 [0,1] or uint16 image to [0,1] for matplotlib."""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 65535.0
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


def mask_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a binary quality mask on a grayscale image.
    Bad pixels (mask == 0) are tinted red.
    Returns an RGB image.
    """
    rgb = np.stack([img, img, img], axis=-1)
    bad = mask == 0
    rgb[bad, 0] = np.clip(rgb[bad, 0] + alpha, 0, 1)   # red channel up
    rgb[bad, 1] = np.clip(rgb[bad, 1] - alpha * 0.5, 0, 1)  # green down
    rgb[bad, 2] = np.clip(rgb[bad, 2] - alpha * 0.5, 0, 1)  # blue down
    return rgb


def coverage(mask: np.ndarray) -> float:
    """Fraction of valid (non-masked) pixels."""
    return float(mask.sum()) / mask.size


# ─────────────────────────────────────────────────────────────────────────────
# Scene-level visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_scene(scene_path: str,
                    scene_name: str,
                    max_frames: int = 9,
                    save_path: str | None = None,
                    show: bool = True) -> dict:
    """
    Render a full visual summary of one scene.

    Returns a dict of per-scene statistics.
    """
    # ── Load HR ──────────────────────────────────────────────────────────────
    hr_raw, hr_mask = highres_image(scene_path)
    hr_disp = normalize_for_display(hr_raw)
    hr_cov  = coverage(hr_mask)

    # ── Load LR frames ───────────────────────────────────────────────────────
    lr_frames, lr_masks = [], []
    for lr_raw, lr_mask_raw in lowres_image_iterator(scene_path):
        lr_frames.append(lr_raw)
        lr_masks.append(lr_mask_raw)
        if len(lr_frames) >= max_frames:
            break

    n_frames = len(lr_frames)
    coverages = [coverage(m) for m in lr_masks]
    best_idx  = int(np.argmax(coverages))

    # ── Layout ───────────────────────────────────────────────────────────────
    # Top row:   LR frames (with mask overlay)
    # Bottom row: HR image | HR mask | coverage bar chart | stats text
    n_cols     = max(n_frames, 4)
    fig_width  = max(16, n_cols * 2.2)
    fig        = plt.figure(figsize=(fig_width, 8), facecolor="#0f1117")

    gs = gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        hspace=0.35,
        wspace=0.08,
        top=0.88, bottom=0.06,
        left=0.03, right=0.97,
    )

    title_color  = "#e8e6df"
    muted_color  = "#9a9890"
    accent_color = "#5b8dee"
    bad_color    = "#e05c5c"

    fig.suptitle(
        f"Proba-V Dataset — Scene: {scene_name}",
        fontsize=14, fontweight="bold",
        color=title_color, y=0.96,
    )

    # ── Top row: LR frames ───────────────────────────────────────────────────
    for i in range(n_cols):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#0f1117")

        if i < n_frames:
            lr_disp = normalize_for_display(lr_frames[i])
            lr_mask = lr_masks[i]
            composite = mask_overlay(lr_disp, lr_mask)
            ax.imshow(composite, interpolation="nearest")

            cov_pct = coverages[i] * 100
            border_col = accent_color if i == best_idx else muted_color
            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(2.0 if i == best_idx else 0.5)

            label = f"LR {i+1}"
            if i == best_idx:
                label += " ★"
            ax.set_title(label, fontsize=8, color=title_color, pad=3)
            ax.set_xlabel(f"{cov_pct:.0f}% valid", fontsize=7, color=muted_color, labelpad=2)
        else:
            # Empty placeholder
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    # ── Bottom-left: HR image ─────────────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[1, :2])
    ax_hr.set_facecolor("#0f1117")
    ax_hr.imshow(hr_disp, cmap="gray", interpolation="nearest")
    ax_hr.set_title("HR ground truth (3× scale)", fontsize=9,
                    color=title_color, pad=4)
    ax_hr.set_xticks([])
    ax_hr.set_yticks([])
    for spine in ax_hr.spines.values():
        spine.set_edgecolor("#4caf7d")
        spine.set_linewidth(1.5)

    # ── Bottom: HR mask ───────────────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[1, 2])
    ax_hm.set_facecolor("#0f1117")
    ax_hm.imshow(hr_mask, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax_hm.set_title("HR mask", fontsize=9, color=title_color, pad=4)
    ax_hm.set_xlabel(f"{hr_cov*100:.1f}% valid", fontsize=7,
                     color=muted_color, labelpad=2)
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])

    # ── Bottom: Coverage bar chart ────────────────────────────────────────────
    if n_cols >= 4:
        ax_bar = fig.add_subplot(gs[1, 3])
        ax_bar.set_facecolor("#181c24")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#2a2e3a")

        bar_cols = [
            accent_color if i == best_idx else bad_color if c < 0.5 else "#6abf8a"
            for i, c in enumerate(coverages)
        ]
        bars = ax_bar.bar(range(n_frames), [c * 100 for c in coverages],
                          color=bar_cols, width=0.75, zorder=2)
        ax_bar.set_ylim(0, 105)
        ax_bar.axhline(50, color=muted_color, linewidth=0.7, linestyle="--", zorder=1)
        ax_bar.set_xticks(range(n_frames))
        ax_bar.set_xticklabels([f"LR{i+1}" for i in range(n_frames)],
                                fontsize=6, color=muted_color)
        ax_bar.set_yticks([0, 50, 100])
        ax_bar.set_yticklabels(["0%", "50%", "100%"], fontsize=6, color=muted_color)
        ax_bar.set_title("LR frame coverage", fontsize=8, color=title_color, pad=4)
        ax_bar.tick_params(colors=muted_color, length=2)
        ax_bar.grid(axis="y", color="#2a2e3a", linewidth=0.5, zorder=0)

    # ── Bottom: Stats text box ────────────────────────────────────────────────
    if n_cols >= 5:
        ax_txt = fig.add_subplot(gs[1, 4:])
    else:
        # Squeeze into last cell
        ax_txt = fig.add_subplot(gs[1, -1])
    ax_txt.set_facecolor("#181c24")
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    for spine in ax_txt.spines.values():
        spine.set_edgecolor("#2a2e3a")

    lr_stack = np.stack([normalize_for_display(f) for f in lr_frames])
    stats_lines = [
        ("Scene",        scene_name),
        ("LR frames",    str(n_frames)),
        ("LR size",      f"{lr_frames[0].shape[1]}×{lr_frames[0].shape[0]} px"),
        ("HR size",      f"{hr_raw.shape[1]}×{hr_raw.shape[0]} px"),
        ("Scale factor", "3×"),
        ("HR coverage",  f"{hr_cov*100:.1f}%"),
        ("Best LR",      f"Frame {best_idx+1} ({coverages[best_idx]*100:.0f}%)"),
        ("LR mean",      f"{lr_stack.mean():.4f}"),
        ("LR std",       f"{lr_stack.std():.4f}"),
        ("HR mean",      f"{normalize_for_display(hr_raw).mean():.4f}"),
    ]

    y_pos = 0.93
    ax_txt.text(0.05, y_pos, "Scene statistics",
                transform=ax_txt.transAxes,
                fontsize=8, fontweight="bold", color=title_color, va="top")
    y_pos -= 0.12

    for key, val in stats_lines:
        ax_txt.text(0.05, y_pos, key,
                    transform=ax_txt.transAxes,
                    fontsize=7, color=muted_color, va="top")
        ax_txt.text(0.95, y_pos, val,
                    transform=ax_txt.transAxes,
                    fontsize=7, color=title_color, va="top", ha="right")
        y_pos -= 0.09

    # ── Legend note ───────────────────────────────────────────────────────────
    fig.text(
        0.03, 0.01,
        "Red overlay = masked (bad) pixels   |   ★ = best LR frame by coverage   |   Green border = HR ground truth",
        fontsize=7, color=muted_color, ha="left",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)

    return {
        "scene":        scene_name,
        "n_frames":     n_frames,
        "hr_coverage":  hr_cov,
        "best_lr_idx":  best_idx,
        "coverages":    coverages,
        "lr_mean":      float(lr_stack.mean()),
        "lr_std":       float(lr_stack.std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level overview
# ─────────────────────────────────────────────────────────────────────────────

def visualize_dataset_overview(data_path: str,
                                channel: str = "RED",
                                n_scenes: int = 5,
                                max_frames: int = 9,
                                save_dir: str | None = None,
                                show: bool = True) -> None:
    """
    Run visualize_scene() for the first n_scenes scenes and
    print a summary table to the console.
    """
    channel_path = os.path.join(data_path, channel)
    if not os.path.exists(channel_path):
        raise FileNotFoundError(f"Channel folder not found: {channel_path}")

    scenes = sorted([
        f for f in os.listdir(channel_path)
        if not f.startswith(".") and os.path.isdir(os.path.join(channel_path, f))
    ])

    print(f"\n{'─'*60}")
    print(f"  Proba-V Dataset Visualizer")
    print(f"  Channel: {channel}  |  Total scenes: {len(scenes)}")
    print(f"  Showing first {min(n_scenes, len(scenes))} scenes")
    print(f"{'─'*60}\n")

    all_stats = []
    for i, scene_name in enumerate(scenes[:n_scenes]):
        scene_path = os.path.join(channel_path, scene_name)
        print(f"[{i+1}/{min(n_scenes, len(scenes))}] Scene: {scene_name}")

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"{channel}_{scene_name}.png")

        stats = visualize_scene(
            scene_path=scene_path,
            scene_name=scene_name,
            max_frames=max_frames,
            save_path=save_path,
            show=show,
        )
        all_stats.append(stats)

    # ── Console summary table ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Summary")
    print(f"{'─'*60}")
    header = f"  {'Scene':<20} {'Frames':>6} {'HR cov':>8} {'Best LR':>8} {'LR mean':>8}"
    print(header)
    print(f"  {'─'*58}")
    for s in all_stats:
        best_cov = s["coverages"][s["best_lr_idx"]] * 100
        print(
            f"  {s['scene']:<20} "
            f"{s['n_frames']:>6} "
            f"{s['hr_coverage']*100:>7.1f}% "
            f"{best_cov:>7.0f}% "
            f"{s['lr_mean']:>8.4f}"
        )
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize Proba-V dataset before model inference.")
    p.add_argument("--data_path",  type=str, required=True,
                   help="Path to the train folder (contains RED/ and NIR/ subdirs)")
    p.add_argument("--channel",    type=str, default="RED", choices=["RED", "NIR"],
                   help="Spectral channel to visualize")
    p.add_argument("--scene_idx",  type=int, default=None,
                   help="If set, visualize only this single scene index")
    p.add_argument("--n_scenes",   type=int, default=5,
                   help="Number of scenes to visualize (ignored if --scene_idx is set)")
    p.add_argument("--max_frames", type=int, default=9,
                   help="Maximum number of LR frames to show per scene")
    p.add_argument("--save_dir",   type=str, default=None,
                   help="Directory to save output PNGs (optional)")
    p.add_argument("--no_show",    action="store_true",
                   help="Don't open matplotlib windows (useful for headless runs)")
    return p.parse_args()


def main():
    args = parse_args()
    show = not args.no_show

    if args.scene_idx is not None:
        # Single scene mode
        channel_path = os.path.join(args.data_path, args.channel)
        scenes = sorted([
            f for f in os.listdir(channel_path)
            if not f.startswith(".") and os.path.isdir(os.path.join(channel_path, f))
        ])
        if args.scene_idx >= len(scenes):
            raise IndexError(f"scene_idx {args.scene_idx} out of range (total: {len(scenes)})")
        scene_name = scenes[args.scene_idx]
        scene_path = os.path.join(channel_path, scene_name)

        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"{args.channel}_{scene_name}.png")

        visualize_scene(
            scene_path=scene_path,
            scene_name=scene_name,
            max_frames=args.max_frames,
            save_path=save_path,
            show=show,
        )
    else:
        # Multi-scene overview
        visualize_dataset_overview(
            data_path=args.data_path,
            channel=args.channel,
            n_scenes=args.n_scenes,
            max_frames=args.max_frames,
            save_dir=args.save_dir,
            show=show,
        )


if __name__ == "__main__":
    main()
