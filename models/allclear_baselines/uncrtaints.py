"""
models/allclear_baselines/uncrtaints.py
Wrapper for the UnCRtainTS cloud-removal model.

UnCRtainTS reference:
    Ebel et al., "UnCRtainTS: Uncertainty Quantification for Cloud Removal
    in Optical Satellite Time Series", CVPR 2023.
    Code: models/allclear_baselines/UnCRtainTS/

Required files
--------------
    <checkpoints_dir>/<exp_name>/conf.json       — model config
    <checkpoints_dir>/<exp_name>/model.pth.tar   — checkpoint

Typical usage
-------------
    model = UnCRtainTS(device="cpu")
    pred_np = model.predict(sample_dict)   # (13, H, W) float32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

_DEFAULT_BASELINE_DIR    = os.path.join(os.path.dirname(__file__), "UnCRtainTS", "model")
_DEFAULT_CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "UnCRtainTS", "model", "checkpoints")


class UnCRtainTS:
    """
    Predict cloud-free Sentinel-2 imagery using the UnCRtainTS model.

    predict() receives the sample_dict from AllClearDataset.load_sample()
    and returns a (13, H, W) float32 numpy array.

    Args:
        baseline_dir:    path to the UnCRtainTS *model* directory
                         (default: models/allclear_baselines/UnCRtainTS/model)
                         Added to sys.path so its src/ package is importable.
        checkpoints_dir: directory containing experiment subfolders
                         (default: models/allclear_baselines/UnCRtainTS/model/checkpoints)
        exp_name:        experiment subfolder name, e.g. "noSAR_1"
        device:          "cpu" or "cuda"
    """

    S2_BANDS = 13

    def __init__(self, baseline_dir=None, checkpoints_dir=None, exp_name="noSAR_1",
                 device="cpu"):
        self.device   = torch.device(device)
        self.exp_name = exp_name

        baseline_dir    = os.path.abspath(baseline_dir    or _DEFAULT_BASELINE_DIR)
        checkpoints_dir = os.path.abspath(checkpoints_dir or _DEFAULT_CHECKPOINTS_DIR)

        # ── make UnCRtainTS source importable ────────────────────────────────────
        if baseline_dir not in sys.path:
            sys.path.insert(0, baseline_dir)

        from src.model_utils import get_model, load_checkpoint
        from parse_args import create_parser
        from src.utils import str2list

        # ── load config ──────────────────────────────────────────────────────────
        conf_path = os.path.join(checkpoints_dir, exp_name, "conf.json")
        with open(conf_path) as f:
            conf = json.load(f)

        parser  = create_parser(mode="test")
        ns      = argparse.Namespace(**conf)
        ns.device = device
        config, _ = parser.parse_known_args(namespace=ns)
        config    = str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
        config.experiment_name = exp_name

        # ── build and load model ─────────────────────────────────────────────────
        self.model = get_model(config).to(self.device)
        load_checkpoint(config, checkpoints_dir, self.model, "model")
        self.model.eval()

        # no SAR = 13 input channels; with SAR = 15 (13 S2 + 2 S1)
        self.num_input_dims = 13 if not conf.get("use_sar", False) else 15

    # ─────────────────────────────────────────────────────────────────────────────

    def predict(self, sample_dict: dict) -> np.ndarray:
        """
        Run UnCRtainTS inference on a single AllClear sample.

        Args:
            sample_dict: dict produced by AllClearDataset.load_sample(), with
                'input_images'    (C, T, H, W) float32 tensor
                'input_cld_shdw'  (2, T, H, W) float32 tensor
                'target'          (C, 1, H, W) float32 tensor  [s2p]
                'time_differences'(T,)         float32 tensor  [days from t=0]

        Returns:
            np.ndarray of shape (13, H, W) — predicted cloud-free S2 image.
        """
        imgs  = sample_dict["input_images"].unsqueeze(0)   # (1, C, T, H, W)
        cld   = sample_dict["input_cld_shdw"].unsqueeze(0) # (1, 2, T, H, W)
        tgt   = sample_dict["target"].unsqueeze(0)         # (1, C, 1, H, W)
        diffs = sample_dict["time_differences"].unsqueeze(0).to(self.device)  # (1, T)

        # (1, C, T, H, W) → (1, T, C, H, W)
        imgs = imgs.permute(0, 2, 1, 3, 4).to(self.device)  # (1, T, C, H, W)

        # Pad missing channels with zeros when SAR data is absent but the
        # checkpoint expects more channels (e.g. multitemporalL2 needs 15).
        C_avail = imgs.shape[2]
        if C_avail < self.num_input_dims:
            pad = torch.zeros(
                imgs.shape[0], imgs.shape[1],
                self.num_input_dims - C_avail,
                imgs.shape[3], imgs.shape[4],
                device=self.device,
            )
            imgs = torch.cat([imgs, pad], dim=2)
        else:
            imgs = imgs[:, :, : self.num_input_dims]
        tgt  = tgt.permute(0, 2, 1, 3, 4)[:, :, : self.S2_BANDS].to(self.device)

        # (1, 2, T, H, W) → combined (1, T, H, W) cloud mask
        masks = torch.clip(cld.sum(dim=1), 0, 1).to(self.device)

        model_inputs = {"A": imgs, "B": tgt, "dates": diffs, "masks": masks}

        with torch.no_grad():
            self.model.set_input(model_inputs)
            self.model.forward()
            self.model.get_loss_G()
            self.model.rescale()
            out = self.model.fake_B          # (1, T_out, C_out, H, W)
            out = out[:, :, : self.S2_BANDS] # keep only mean predictions

        # Return first output timestep as (13, H, W)
        return out[0, 0].cpu().numpy()
