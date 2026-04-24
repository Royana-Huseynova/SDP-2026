import numpy as np
from datasets.probav.io import lowres_image_iterator
from datasets.probav.transforms import bicubic_upscaling

def baseline_upscale(path: str) -> np.ndarray:
    """
    ESA competition baseline: bicubic upscale of clearest LR frames.
    (Kept for backwards compatibility)
    """
    clearance = {}
    for (l, c) in lowres_image_iterator(path, img_as_float=True):
        clearance.setdefault(c.sum(), []).append(l)

    imgs = max(clearance.items(), key=lambda i: i[0])[1]
    sr = np.mean([bicubic_upscaling(i) for i in imgs], axis=0)
    return sr

class BicubicModel:
    """
    Wrapper class to make the bicubic baseline compatible 
    with the pipeline's .predict() interface.
    """
    def predict(self, lr_input: np.ndarray) -> np.ndarray:
        """
        Takes the input tensor (T, 128, 128) and performs 
        bicubic upscaling on each frame, then averages the result.
        """
        # Perform bicubic upscaling on every frame in the stack (T, H, W)
        upscaled_frames = [bicubic_upscaling(frame) for frame in lr_input]
        
        # Calculate the mean across the time dimension
        sr = np.mean(upscaled_frames, axis=0)
        return sr