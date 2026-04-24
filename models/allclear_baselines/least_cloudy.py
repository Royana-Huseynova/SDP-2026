import numpy as np
import torch


class LeastCloudy:
    """
    Select the temporally least-cloudy frame as the cloud-removal output.

    Requires no learning. For each sample the frame with the lowest total
    cloud + shadow pixel count is returned as-is.

    predict() interface matches the unified pipeline: receives the sample_dict
    produced by AllClearDataset.load_sample() and returns a (C, H, W) float32
    numpy array containing the S2 bands of the chosen frame.
    """

    def predict(self, sample_dict: dict) -> np.ndarray:
        """
        Args:
            sample_dict: dict with at least
                'input_images'   (C, T, H, W) float32 tensor
                'input_cld_shdw' (2, T, H, W) float32 tensor

        Returns:
            np.ndarray of shape (13, H, W) — the S2 bands of the least
            cloudy input frame.
        """
        images  = sample_dict["input_images"]   # (C, T, H, W)
        cld_shdw = sample_dict["input_cld_shdw"]  # (2, T, H, W)

        # Cloud cover per timestep: sum over channels and spatial dims → (T,)
        cloudiness = cld_shdw.sum(dim=(0, 2, 3))
        least_cloudy_t = int(cloudiness.argmin())

        # Return only the S2 channels (first 13), spatial dims intact
        output = images[:13, least_cloudy_t, :, :]   # (13, H, W)
        return output.numpy()
