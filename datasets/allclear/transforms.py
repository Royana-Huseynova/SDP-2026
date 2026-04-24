import torch


def preprocess_s2(image):
    """Clip Sentinel-2 TOA DN to [0, 10000] and scale to [0, 1]."""
    image = torch.clip(image, 0, 10000) / 10000.0
    return torch.nan_to_num(image, nan=0.0)


def preprocess_s1(image, mode="default"):
    """
    Normalize Sentinel-1 SAR dB values to [0, 1].

    VV channel: shifted by +25 dB, clipped to [0, 25], then divided by 25.
    VH channel: shifted by +32.5 dB, clipped to [0, 32.5], then divided by 32.5.
    NaN pixels set to -1 (indicating no data).
    """
    image = image.clone()
    if mode == "default":
        image[image < -40] = -40
        image[0] = torch.clip(image[0] + 25.0, 0.0, 25.0) / 25.0
        image[1] = torch.clip(image[1] + 32.5, 0.0, 32.5) / 32.5
        image = torch.nan_to_num(image, nan=-1.0)
    return image


def preprocess_cld_shdw(image):
    """Fill NaN in cloud/shadow masks with 1 (treat missing as cloudy)."""
    return torch.nan_to_num(image, nan=1.0)


def cld_shdw_to_valid_mask(cld_shdw):
    """
    Convert a (2, H, W) cloud+shadow float mask to a (H, W) bool valid mask.

    Returns True where the pixel is clear (neither cloudy nor shadowed).
    """
    combined = (cld_shdw[0] + cld_shdw[1]) > 0
    return ~combined
