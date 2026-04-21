"""
Loads the pretrained DeepSUM checkpoint and runs inference.
Follows predict_test() from DeepSUM_network.py exactly.

Key facts from source:
  - Input normalized: (x - mu) / sigma  where x is in 16-bit scale
  - fill_coeff computed from LR quality masks
  - feed_dict only needs: x and fill_coeff
  - Output tensor: add_18:0 — outputs at LR size (128x128)
  - Output de-normalized: (logits * sigma_rescaled) + mu
  - Brightness corrected to match LR mean
  - Bicubic upsample to HR size (384x384)

Config values from DeepSUM_config_NIR.json:
  mu             = 7754.664
  sigma          = 2316.3408
  sigma_rescaled = 596.982
"""

import os
import numpy as np
import skimage.transform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

MU             = 7754.664    # NIR mu
SIGMA          = 2316.3408   # NIR sigma
SIGMA_RESCALED = 596.982     # NIR sigma_rescaled
T_IN           = 9


def compute_fill_coeff(masks: np.ndarray) -> np.ndarray:
    """
    Compute fill coefficients from LR quality masks.
    Copied from predict_test() in DeepSUM_network.py.
    """
    T, H, W = masks.shape
    masks = masks.astype(bool)

    masks_exp = masks[:, :, :, np.newaxis]
    masks_b   = masks_exp[np.newaxis, :, :, :, :]

    sh = masks_b.shape
    fill_coeff = np.ones([sh[0], sh[1], sh[1], sh[2], sh[3], sh[4]], dtype=bool)

    for i in range(0, 9):
        fill_coeff[:, :, i] = np.expand_dims(masks_b[:, i], axis=1)

    for i in range(0, 9):
        for j in range(i + 1, 9):
            rows_indexes = [k for k in range(0, 9) if k != j]
            fill_coeff[:, rows_indexes, j] = (
                fill_coeff[:, rows_indexes, j] *
                np.expand_dims(1 - masks_b[:, i], axis=1)
            )

    for i in range(1, 9):
        fill_coeff[:, i, 0:i] = (
            fill_coeff[:, i, 0:i] *
            np.expand_dims(1 - masks_b[:, i], axis=1)
        )

    f = np.sum(fill_coeff, axis=2)
    fill_coeff[:, range(9), range(9), :, :, :] = (
        fill_coeff[:, range(9), range(9), :, :, :] +
        np.logical_not(f)[:, range(9), :, :, :]
    )

    return fill_coeff.astype(np.float32)


class DeepSUMModel:
    def __init__(self,
                 checkpoint_dir: str = "benchmark/checkpoints/DeepSUM_NIR_lr_5e-06_bsize_8",
                 t_in: int = None,
                 upscale: int = 3,
                 mu: float = None,
                 sigma: float = None,
                 sigma_rescaled: float = None):

        t_in           = t_in           if t_in           is not None else T_IN
        mu             = mu             if mu             is not None else MU
        sigma          = sigma          if sigma          is not None else SIGMA
        sigma_rescaled = sigma_rescaled if sigma_rescaled is not None else SIGMA_RESCALED

        self.t_in           = t_in
        self.upscale        = upscale
        self.mu             = mu
        self.sigma          = sigma
        self.sigma_rescaled = sigma_rescaled

        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
        print(f"[DeepSUM] Loading checkpoint: {ckpt}")

        self.graph   = tf.compat.v1.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph)

        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta')
            saver.restore(self.session, ckpt)

            self.ph_x          = self.graph.get_tensor_by_name('x:0')
            self.ph_fill_coeff = self.graph.get_tensor_by_name('fill_coeff:0')
            self.output        = self.graph.get_tensor_by_name('add_18:0')

        print(f"[DeepSUM] Model ready.")

    def predict(self, lr_stack: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Run super-resolution on one scene.

        Args:
            lr_stack: (T, H, W) float32 in [0, 1]
            masks:    (T, H, W) bool — True = valid pixel (optional)

        Returns:
            sr: (H*3, W*3) float32 in [0, 1]
        """
        if lr_stack.ndim != 3:
            raise ValueError(f"Expected (T, H, W), got {lr_stack.shape}")

        T, H, W = lr_stack.shape

        if T > self.t_in:
            lr    = lr_stack[:self.t_in].astype(np.float32)
            masks = masks[:self.t_in] if masks is not None else None
        elif T < self.t_in:
            pad = np.zeros((self.t_in - T, H, W), dtype=np.float32)
            lr  = np.concatenate([lr_stack.astype(np.float32), pad], axis=0)
            if masks is not None:
                mpad  = np.zeros((self.t_in - T, H, W), dtype=bool)
                masks = np.concatenate([masks, mpad], axis=0)
        else:
            lr = lr_stack.astype(np.float32)

        if masks is None:
            masks = np.ones((self.t_in, H, W), dtype=bool)

        # Record LR mean for brightness correction later
        lr_mean = float(lr.mean())

        # ── Normalize: [0,1] → 16-bit → z-score ──────────────────────────────
        lr_16bit = lr * 65535.0
        lr_norm  = (lr_16bit - self.mu) / self.sigma
        x_feed   = lr_norm[np.newaxis, :, :, :, np.newaxis]

        # ── Compute fill_coeff from masks ─────────────────────────────────────
        fill_coeff = compute_fill_coeff(masks.astype(bool))

        # ── Run inference ─────────────────────────────────────────────────────
        with self.graph.as_default():
            logits = self.session.run(
                self.output,
                feed_dict={
                    self.ph_x:          x_feed,
                    self.ph_fill_coeff: fill_coeff,
                }
            )

        # ── De-normalize → 16-bit → [0,1] ────────────────────────────────────
        sr_16bit = (np.squeeze(logits) * self.sigma_rescaled) + self.mu
        sr       = (sr_16bit / 65535.0).astype(np.float32)

        # ── Brightness correction: shift SR mean to match LR mean ─────────────
        sr_mean           = float(sr.mean())
        brightness_offset = lr_mean - sr_mean
        sr = sr + brightness_offset

        # ── Bicubic upsample from LR (128x128) to HR (384x384) ────────────────
        sr_hr = skimage.transform.rescale(
            sr, scale=self.upscale, order=3,
            mode='edge', anti_aliasing=False
        ).astype(np.float32)

        sr_hr = np.clip(sr_hr, 0.0, 1.0)

        return sr_hr

    def close(self):
        self.session.close()
        print("[DeepSUM] Session closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    checkpoint_dir = "benchmark/checkpoints/DeepSUM_NIR_lr_5e-06_bsize_8"

    print("=" * 60)
    print("DeepSUM Model — Quick Test")
    print("=" * 60)

    with DeepSUMModel(checkpoint_dir=checkpoint_dir) as model:
        dummy_lr    = np.random.rand(9, 128, 128).astype(np.float32)
        dummy_masks = np.ones((9, 128, 128), dtype=bool)
        print(f"\nInput shape: {dummy_lr.shape}")
        sr = model.predict(dummy_lr, dummy_masks)
        print(f"Output shape: {sr.shape}")
        print(f"Output range: {sr.min():.4f} – {sr.max():.4f}")
        if sr.shape == (384, 384):
            print("\n✓ Model working correctly!")
        else:
            print(f"\n? Output shape: {sr.shape}")