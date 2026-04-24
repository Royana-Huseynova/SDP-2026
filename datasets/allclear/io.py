import json
import numpy as np
import rasterio as rs
import torch


def load_geotiff(fpath, channels=None, size=None):
    """
    Load a GeoTIFF file into a float32 tensor (C, H, W).

    Args:
        fpath:    path to the .tif file
        channels: list of 1-based band indices to read, or None for all bands
        size:     (W, H) tuple for center-cropping, or None for full image

    Returns:
        torch.Tensor of shape (C, H, W), dtype float32
    """
    with rs.open(fpath) as src:
        if size is not None:
            w, h = src.width, src.height
            ox = (w - size[0]) // 2
            oy = (h - size[1]) // 2
            window = rs.windows.Window(ox, oy, size[0], size[1])
            data = src.read(channels, window=window) if channels else src.read(window=window)
        else:
            data = src.read(channels) if channels else src.read()
    return torch.from_numpy(data.astype(np.float32))


def load_metadata(json_path):
    """
    Load AllClear dataset metadata from a JSON file.

    The JSON maps data_id strings to dicts containing 'roi', sensor file lists,
    and optionally a 'target' entry for seq2point mode.

    Returns:
        dict keyed by data_id
    """
    with open(json_path) as f:
        return json.load(f)
