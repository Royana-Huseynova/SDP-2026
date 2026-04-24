import numpy as np
import torch


class Mosaicing:
    """
    Pixel-level temporal mosaicing: average the clear observations at each
    pixel across all input timesteps.

    For each spatial position, only pixels that are clear (not flagged by the
    cloud or shadow mask) contribute to the average.  Pixels that are cloudy
    in every frame fall back to 0.5 (mid-range).

    predict() interface matches the unified pipeline: receives the sample_dict
    produced by AllClearDataset.load_sample() and returns a (C, H, W) float32
    numpy array of the mosaiced S2 output.
    """

    def predict(self, sample_dict: dict) -> np.ndarray:
        """
        Args:
            sample_dict: dict with at least
                'input_images'   (C, T, H, W) float32 tensor
                'input_cld_shdw' (2, T, H, W) float32 tensor

        Returns:
            np.ndarray of shape (13, H, W) — mosaiced S2 image.
        """
        images   = sample_dict["input_images"]   # (C, T, H, W)
        cld_shdw = sample_dict["input_cld_shdw"]  # (2, T, H, W)

        s2 = images[:13]  # (13, T, H, W) — S2 channels are always first

        # Combined cloud+shadow mask: (1, T, H, W), 1 = cloudy or shadowed
        cloud_mask = torch.clip(cld_shdw.sum(dim=0, keepdim=True), 0, 1)
        clear_mask = 1.0 - cloud_mask  # (1, T, H, W), 1 = clear

        # Weighted sum over the temporal dimension
        sum_clear_pixels = (s2 * clear_mask).sum(dim=1)          # (13, H, W)
        sum_clear_views  = clear_mask.sum(dim=1).clamp(min=1.0)  # (1, H, W)
        mosaic = sum_clear_pixels / sum_clear_views               # (13, H, W)

        # Where no clear view exists set pixels to mid-range (0.5)
        no_clear = (clear_mask.sum(dim=1) == 0)  # (1, H, W)
        mosaic[no_clear.expand_as(mosaic)] = 0.5

        return mosaic.numpy()
