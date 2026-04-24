import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ==============================================================
# IMAGE I/O HELPERS
# ==============================================================

def highres_image(scene_path):
    """Load the high-resolution image and its mask (SM.png)."""
    hr_path = os.path.join(scene_path, "HR.png")
    sm_path = os.path.join(scene_path, "SM.png")

    hr = np.array(Image.open(hr_path)).astype(np.float32) / 65535.0
    # If SM.png doesn't exist, we default to a mask of all True values
    mask = np.array(Image.open(sm_path)).astype(bool) if os.path.exists(sm_path) else np.ones(hr.shape, dtype=bool)

    return hr, mask

def lowres_image_iterator(scene_path):
    """Yield (lr_image, quality_mask) pairs for all LR/QM files in a scene."""
    lr_files = sorted([f for f in os.listdir(scene_path) if f.startswith("LR") and f.endswith(".png")])
    qm_files = sorted([f for f in os.listdir(scene_path) if f.startswith("QM") and f.endswith(".png")])

    for lr_file, qm_file in zip(lr_files, qm_files):
        lr = np.array(Image.open(os.path.join(scene_path, lr_file))).astype(np.float32) / 65535.0
        qm = np.array(Image.open(os.path.join(scene_path, qm_file))).astype(bool)
        yield lr, qm

# ==============================================================
# DATASET CLASS
# ==============================================================

class ProbaVDataset(Dataset):
    def __init__(self, base_path, channel="RED", max_t=9):
        self.base_path = os.path.join(base_path, channel)
        self.max_t = max_t

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Folder not found: {self.base_path}")

        self.scenes = sorted([
            f for f in os.listdir(self.base_path)
            if not f.startswith('.') and os.path.isdir(os.path.join(self.base_path, f))
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.base_path, scene_name)

        # Load high-resolution image and the HR mask
        hr, hr_mask = highres_image(scene_path)
        hr_array = hr.astype(np.float32)

        # Load low-resolution frames
        lr_frames = []
        lr_masks = []
        for lr, mask in lowres_image_iterator(scene_path):
            lr_frames.append(lr.astype(np.float32))
            lr_masks.append(mask)

        lr_stack   = np.stack(lr_frames, axis=0)
        mask_stack = np.stack(lr_masks,  axis=0)

        # Pad or trim to fixed T
        T = lr_stack.shape[0]
        if T < self.max_t:
            pad  = np.zeros((self.max_t - T, *lr_stack.shape[1:]),   dtype=np.float32)
            mpad = np.zeros((self.max_t - T, *mask_stack.shape[1:]), dtype=bool)
            lr_stack   = np.concatenate([lr_stack,   pad],  axis=0)
            mask_stack = np.concatenate([mask_stack, mpad], axis=0)
        elif T > self.max_t:
            lr_stack   = lr_stack[:self.max_t]
            mask_stack = mask_stack[:self.max_t]

        # Return 5 items: lr, hr, lr_mask, hr_mask, name
        return lr_stack, hr_array, mask_stack, hr_mask, scene_name

    def load_sample(self, idx=0):
        return self.__getitem__(idx)