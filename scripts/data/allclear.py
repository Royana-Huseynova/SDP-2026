"""
AllClear dataset helper functions.
"""

from pathlib import Path
from typing import List, Tuple

from utils import read_rgb  # uses stretch etc. internally


def find_tiff_files(data_root: str | Path) -> List[Path]:
    """
    Recursively find all .tif files under data_root.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    files = sorted(root.rglob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif files found under {root}")

    return files


def load_batch(
    data_root: str | Path,
    start: int = 1,
    count: int = 10,
):
    """
    Load a batch of tiles from AllClear as uint8 RGB images.

    Returns:
        imgs           : list of H*W*3 uint8 arrays
        paths          : list of Path objects (same length as imgs)
        global_indices : list of 1-based indices corresponding to each path
        skipped_empty  : number of tiles skipped as empty/nodata
        total_files    : total number of .tif files under data_root
    """
    import numpy as np

    files = find_tiff_files(data_root)
    total_files = len(files)

    # Convert 1-based [start, start+count-1] into 0-based slice
    start_idx = max(0, start - 1)
    end_idx = min(total_files, start_idx + count)
    batch = files[start_idx:end_idx]

    imgs = []
    used_paths: List[Path] = []
    global_indices: List[int] = []
    skipped_empty = 0

    for global_idx, path in enumerate(batch, start=start):
        try:
            rgb = read_rgb(path)
            if rgb is None:
                skipped_empty += 1
                print(f"⚠️ Skipped (empty/nodata): {path.name}")
                continue

            img8 = (np.clip(rgb, 0, 1) * 255).astype("uint8")
            imgs.append(img8)
            used_paths.append(path)
            global_indices.append(global_idx)

        except Exception as e:
            skipped_empty += 1
            print(f"⚠️ Skipped {path.name}: {e}")

    return imgs, used_paths, global_indices, skipped_empty, total_files


def save_pngs(
    imgs,
    paths: List[Path],
    global_indices,
    output_dir: str | Path,
):
    """
    Save a list of RGB uint8 images to disk with the format:
        {index:05d}_{original_stem}.png
    """
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for idx, img, path in zip(global_indices, imgs, paths):
        out_path = out / f"{idx:05d}_{path.stem}.png"
        Image.fromarray(img).save(out_path)
        print(f"✅ Saved: {out_path}")
