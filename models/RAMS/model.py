"""

RAMS inference wrapper for the SDP-2026 benchmark.
Loads pretrained RAMS checkpoint and runs inference on a single scene.

Based on: https://github.com/EscVM/RAMS
Paper: "Multi-Image Super Resolution of Remotely Sensed Images Using
        Residual Attention Deep Neural Networks"

Key facts:
  - Input shape:  (1, H, W, T) — uint16 raw values, T=9 frames
  - Output shape: (1, H*3, W*3, 1) — uint16 raw values
  - Normalization: MEAN=7433.6436, STD=2353.0723 (handled inside network)
  - Checkpoints:  external/RAMS/ckpt/NIR_RAMS or RED_RAMS
  - Network params: filters=32, kernel_size=3, channels=9, r=8, N=12, scale=3
"""

import sys
import os
import numpy as np
import tensorflow as tf

# ── Add RAMS utils to path ────────────────────────────────────────────────────
RAMS_UTILS_DIR = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(RAMS_UTILS_DIR))

from network import RAMS
from prediction import predict_tensor, predict_tensor_permute

# ── Network hyperparameters (must match checkpoint) ───────────────────────────
SCALE       = 3
FILTERS     = 32
KERNEL_SIZE = 3
CHANNELS    = 9
R           = 8
N           = 12


class RAMSModel:
    def __init__(self,
                 band: str = 'NIR',
                 checkpoint_dir: str = None,
                 t_in: int = None):
        """
        Load pretrained RAMS model.

        Parameters
        ----------
        band : str
            'NIR' or 'RED'
        checkpoint_dir : str
            Path to checkpoint directory. Defaults to external/RAMS/ckpt/{band}_RAMS
        t_in : int
            Number of input frames. Defaults to 9.
        """
        self.band   = band
        self.t_in   = t_in if t_in is not None else CHANNELS
        self.scale  = SCALE

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(__file__), 'ckpt', f'{band}_RAMS'
            )
        checkpoint_dir = os.path.abspath(checkpoint_dir)

        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        print(f"[RAMS] Building network ({band} band)...")
        self.model = RAMS(
            scale=SCALE,
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            channels=self.t_in,
            r=R,
            N=N
        )

        print(f"[RAMS] Loading checkpoint from: {checkpoint_dir}")
        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            psnr=tf.Variable(1.0),
            model=self.model
        )
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
        checkpoint.restore(ckpt).expect_partial()
        print(f"[RAMS] Loaded: {ckpt}")

    def predict(self, lr_stack: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Run super-resolution on one scene.

        Parameters
        ----------
        lr_stack : np.ndarray
            Shape (T, H, W) float32 in [0, 1]
        masks : np.ndarray
            Shape (T, H, W) bool — True = valid pixel (optional, not used by RAMS)

        Returns
        -------
        sr : np.ndarray
            Shape (H*3, W*3) float32 in [0, 1]
        """
        if lr_stack.ndim != 3:
            raise ValueError(f"Expected (T, H, W), got {lr_stack.shape}")

        T, H, W = lr_stack.shape

        # ── Pad or trim to t_in frames ────────────────────────────────────────
        if T > self.t_in:
            lr = lr_stack[:self.t_in]
        elif T < self.t_in:
            pad = np.zeros((self.t_in - T, H, W), dtype=np.float32)
            lr  = np.concatenate([lr_stack, pad], axis=0)
        else:
            lr = lr_stack.copy()

        # ── Convert [0,1] → uint16 scale (RAMS expects raw 16-bit values) ─────
        lr_16bit = (lr.astype(np.float32) * 65535.0)

        # ── Reshape: (T, H, W) → (1, H, W, T) ───────────────────────────────
        x = lr_16bit.transpose(1, 2, 0)[np.newaxis, ...]  # (1, 128, 128, 9)

        # ── Run inference ─────────────────────────────────────────────────────
        sr_raw = predict_tensor(self.model, x)  # (1, 384, 384, 1)

        # ── Convert back to [0, 1] ────────────────────────────────────────────
        sr = sr_raw.numpy()[0, :, :, 0] / 65535.0
        sr = np.clip(sr, 0.0, 1.0).astype(np.float32)

        return sr

    def predict_plus(self, lr_stack: np.ndarray, masks: np.ndarray = None, n_permut: int = 20) -> np.ndarray:
        """
        RAMS+ prediction — ensemble with frame shuffling for better accuracy.
        Slower but more accurate than predict().

        Parameters
        ----------
        lr_stack : np.ndarray
            Shape (T, H, W) float32 in [0, 1]
        n_permut : int
            Number of permutations for ensemble (default 20)

        Returns
        -------
        sr : np.ndarray
            Shape (H*3, W*3) float32 in [0, 1]
        """
        if lr_stack.ndim != 3:
            raise ValueError(f"Expected (T, H, W), got {lr_stack.shape}")

        T, H, W = lr_stack.shape

        if T > self.t_in:
            lr = lr_stack[:self.t_in]
        elif T < self.t_in:
            pad = np.zeros((self.t_in - T, H, W), dtype=np.float32)
            lr  = np.concatenate([lr_stack, pad], axis=0)
        else:
            lr = lr_stack.copy()

        lr_16bit = (lr.astype(np.float32) * 65535.0)
        x = lr_16bit.transpose(1, 2, 0)  # (H, W, T) — no batch dim for permute

        sr_raw = predict_tensor_permute(self.model, x, n_ens=n_permut)  # (1, 384, 384, 1)

        sr = sr_raw[0, :, :, 0] / 65535.0
        sr = np.clip(sr, 0.0, 1.0).astype(np.float32)

        return sr


if __name__ == "__main__":
    print("=" * 60)
    print("RAMS Model — Quick Test")
    print("=" * 60)

    model = RAMSModel(band='NIR')

    dummy_lr    = np.random.rand(9, 128, 128).astype(np.float32)
    dummy_masks = np.ones((9, 128, 128), dtype=bool)

    print(f"\nInput shape:  {dummy_lr.shape}")
    sr = model.predict(dummy_lr, dummy_masks)
    print(f"Output shape: {sr.shape}")
    print(f"Output range: {sr.min():.4f} – {sr.max():.4f}")

    if sr.shape == (384, 384):
        print("\n✓ RAMS model working correctly!")
    else:
        print(f"\n? Unexpected output shape: {sr.shape}")