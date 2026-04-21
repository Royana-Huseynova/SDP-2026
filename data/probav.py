import os
import numpy as np
from torch.utils.data import Dataset
from data.io import highres_image, lowres_image_iterator


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

        # Load HR — skimage already returns float64 in [0, 1]
        hr, hr_mask = highres_image(scene_path)
        hr_array = hr.astype(np.float32)               # (384, 384)

        # Load all LR frames — preserve the full temporal stack
        lr_frames = []
        lr_masks = []
        for lr, mask in lowres_image_iterator(scene_path):
            lr_frames.append(lr.astype(np.float32))    # already [0, 1], no /65535
            lr_masks.append(mask)

        lr_stack   = np.stack(lr_frames, axis=0)       # (T, 128, 128)
        mask_stack = np.stack(lr_masks,  axis=0)       # (T, 128, 128) bool

        # Pad or trim to fixed T so batches have consistent shape
        T = lr_stack.shape[0]
        if T < self.max_t:
            pad = np.zeros((self.max_t - T, *lr_stack.shape[1:]), dtype=np.float32)
            lr_stack = np.concatenate([lr_stack, pad], axis=0)
            mpad = np.zeros((self.max_t - T, *mask_stack.shape[1:]), dtype=bool)
            mask_stack = np.concatenate([mask_stack, mpad], axis=0)
        elif T > self.max_t:
            lr_stack   = lr_stack[:self.max_t]
            mask_stack = mask_stack[:self.max_t]

        return lr_stack, hr_array, mask_stack, scene_name