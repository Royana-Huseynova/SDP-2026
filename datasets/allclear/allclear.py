import os
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import Dataset

from .io import load_geotiff, load_metadata
from .transforms import (
    preprocess_s2,
    preprocess_s1,
    preprocess_cld_shdw,
    cld_shdw_to_valid_mask,
)

# Default band indices per sensor (1-based, as used by rasterio)
_CHANNELS = {
    "s2_toa":   list(range(1, 14)),   # 13 Sentinel-2 TOA bands
    "s1":        [1, 2],               # VV, VH SAR bands
    "landsat8":  list(range(1, 12)),   # 11 Landsat-8 bands
    "landsat9":  list(range(1, 12)),   # 11 Landsat-9 bands
    "cld_shdw":  [2, 5],              # cloud (band 2) and shadow (band 5)
    "dw":        [1],                  # Dynamic World LULC class
}


def _temporal_align(main_timestamps, aux_ts, max_diff=2):
    """
    Return the index in main_timestamps closest to aux_ts, if within max_diff days.
    Returns None when no close-enough match exists.
    """
    diffs = [abs((ts - aux_ts).total_seconds() / 86400.0) for ts in main_timestamps]
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] <= max_diff else None


class AllClearDataset(Dataset):
    """
    Sentinel-2 cloud removal dataset from AllClear.

    Each sample is a temporal sequence of cloudy S2 observations (plus optional
    auxiliary sensors) together with a cloud-free target S2 image.

    Directory / metadata layout expected by the JSON::

        {
          "<data_id>": {
            "roi": ["<roi_id>", [lat, lon]],
            "target": [["<timestamp>", "<path_to_tif>"]],
            "s2_toa": [["<timestamp>", "<path>"], ...],
            "s1":     [["<timestamp>", "<path>"], ...]   # optional
          },
          ...
        }

    load_sample() returns a 5-tuple that matches the unified pipeline API::

        (sample_dict, target_np, lr_masks_np, hr_mask_np, name)

        sample_dict  – dict of tensors consumed by AllClear models:
                         "input_images"    (C, T, H, W) float32
                         "input_cld_shdw"  (2, T, H, W) float32
                         "target"          (C, 1, H, W) float32  [s2p]
                         "target_cld_shdw" (2, 1, H, W) float32  [s2p]
                         "roi"             str
                         "data_id"         str
        target_np    – (C, H, W) float32 np.ndarray  ground-truth S2 image
        lr_masks_np  – (T, H, W) bool    np.ndarray  per-frame cloud masks
                         True = pixel is cloudy or shadowed
        hr_mask_np   – (H, W)    bool    np.ndarray  target valid pixels
                         True = clear (not cloudy / not shadowed)
        name         – str  data_id
    """

    def __init__(
        self,
        json_path,
        selected_rois="all",
        main_sensor="s2_toa",
        aux_sensors=None,
        tx=3,
        center_crop_size=(256, 256),
        target_mode="s2p",
        s2_toa_channels=None,
        max_diff=2,
    ):
        if aux_sensors is None:
            aux_sensors = []

        self.main_sensor = main_sensor
        self.aux_sensors = aux_sensors
        self.sensors = [main_sensor] + aux_sensors
        self.tx = tx
        self.crop = center_crop_size
        self.target_mode = target_mode
        self.max_diff = max_diff

        self.channels = dict(_CHANNELS)
        if s2_toa_channels is not None:
            self.channels["s2_toa"] = s2_toa_channels

        raw = load_metadata(json_path)
        if selected_rois == "all":
            self.dataset = raw
        else:
            self.dataset = {
                k: v for k, v in raw.items() if v["roi"][0] in selected_rois
            }
        self.ids = list(self.dataset.keys())

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, fpath, sensor):
        """Load and preprocess a single GeoTIFF for the given sensor."""
        image = load_geotiff(fpath, channels=self.channels[sensor], size=self.crop)
        if sensor in ("s2_toa", "landsat8", "landsat9"):
            return preprocess_s2(image)
        if sensor == "s1":
            return preprocess_s1(image)
        if sensor == "cld_shdw":
            return preprocess_cld_shdw(image)
        return image

    @staticmethod
    def _cld_path(s2_path):
        return s2_path.replace("s2_toa", "cld_shdw")

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data_id = self.ids[idx]
        sample  = self.dataset[data_id]
        roi     = sample["roi"][0]

        H, W = self.crop

        # ── total channel count: main sensor first, then aux in order ─────────
        C_main = len(self.channels[self.main_sensor])
        C_aux  = sum(len(self.channels[s]) for s in self.aux_sensors)
        C_total = C_main + C_aux

        # Buffers initialised to 1 (clouds mask: all-cloudy; images: max value).
        # Padded / missing timesteps stay as ones so cloud-removal models treat
        # them as unusable observations.
        buf     = torch.ones(C_total, self.tx, H, W)
        cld_buf = torch.ones(2, self.tx, H, W)

        # ── main sensor ───────────────────────────────────────────────────────
        main_frames: list[tuple[datetime, torch.Tensor]] = []
        cld_frames:  list[tuple[datetime, torch.Tensor]] = []

        for ts_str, fpath in sample[self.main_sensor]:
            ts  = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            img = self._load(fpath, self.main_sensor)
            main_frames.append((ts, img))

            cld_fpath = self._cld_path(fpath)
            if os.path.exists(cld_fpath):
                cld = self._load(cld_fpath, "cld_shdw")
            else:
                cld = torch.ones(2, H, W)  # treat missing mask as fully cloudy
            cld_frames.append((ts, cld))

        main_frames.sort(key=lambda x: x[0])
        cld_frames.sort(key=lambda x: x[0])
        main_timestamps = [ts for ts, _ in main_frames]

        # Days from first timestamp, padded to tx with the last known value
        if main_timestamps:
            t0 = main_timestamps[0]
            raw_diffs = [round((ts - t0).total_seconds() / 86400) for ts in main_timestamps[: self.tx]]
        else:
            raw_diffs = []
        while len(raw_diffs) < self.tx:
            raw_diffs.append(raw_diffs[-1] if raw_diffs else 0)
        time_diffs = torch.tensor(raw_diffs, dtype=torch.float32)  # (T,)

        for t_idx, (_, img) in enumerate(main_frames[: self.tx]):
            buf[:C_main, t_idx] = img
        for t_idx, (_, cld) in enumerate(cld_frames[: self.tx]):
            cld_buf[:, t_idx] = cld

        # ── auxiliary sensors (temporally aligned to main) ────────────────────
        ch_offset = C_main
        for sensor in self.aux_sensors:
            c_len = len(self.channels[sensor])
            if sensor in sample:
                for ts_str, fpath in sample[sensor]:
                    ts    = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    t_idx = _temporal_align(main_timestamps, ts, self.max_diff)
                    if t_idx is None or t_idx >= self.tx:
                        continue
                    img = self._load(fpath, sensor)
                    buf[ch_offset : ch_offset + c_len, t_idx] = img
            ch_offset += c_len

        # ── target ────────────────────────────────────────────────────────────
        if self.target_mode == "s2p":
            if "target" not in sample:
                raise ValueError(f"No 'target' key in sample '{data_id}' for s2p mode.")
            ts_str, fpath = sample["target"][0]
            target_img = self._load(fpath, self.main_sensor)          # (C, H, W)
            cld_fpath  = self._cld_path(fpath)
            target_cld = (
                self._load(cld_fpath, "cld_shdw")
                if os.path.exists(cld_fpath)
                else torch.zeros(2, H, W)    # no clouds on the clear target
            )
        else:  # s2s — use the input sequence as both input and target
            target_img = buf[:C_main]        # (C, T, H, W)
            target_cld = cld_buf             # (2, T, H, W)

        # ── assemble sample dict (consumed by AllClear models) ────────────────
        if self.target_mode == "s2p":
            target_dict = target_img.unsqueeze(1)     # (C, 1, H, W)
            target_cld_dict = target_cld.unsqueeze(1) # (2, 1, H, W)
        else:
            target_dict     = target_img
            target_cld_dict = target_cld

        sample_dict = {
            "input_images":     buf,             # (C, T, H, W)
            "input_cld_shdw":   cld_buf,         # (2, T, H, W)
            "target":           target_dict,
            "target_cld_shdw":  target_cld_dict,
            "time_differences": time_diffs,      # (T,)  days from first frame
            "roi":     roi,
            "data_id": data_id,
        }

        # ── numpy outputs for the pipeline evaluate step ──────────────────────
        if self.target_mode == "s2p":
            target_np   = target_img.numpy()                             # (C, H, W)
            hr_mask_np  = cld_shdw_to_valid_mask(target_cld).numpy()    # (H, W) bool
        else:
            # s2s: take first frame as representative target
            target_np  = buf[:C_main, 0].numpy()
            hr_mask_np = cld_shdw_to_valid_mask(cld_buf[:, 0]).numpy()

        lr_masks_np = ((cld_buf[0] + cld_buf[1]) > 0).numpy()           # (T, H, W) bool

        return sample_dict, target_np, lr_masks_np, hr_mask_np, data_id

    def load_sample(self, idx=0):
        return self.__getitem__(idx)
