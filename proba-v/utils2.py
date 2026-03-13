import numpy as np

def normalize(img):
    """Normalizes image to [0, 1] range."""
    img = img.astype(np.float32)
    denom = img.max() - img.min()
    if denom == 0:
        return np.zeros_like(img)
    return (img - img.min()) / denom