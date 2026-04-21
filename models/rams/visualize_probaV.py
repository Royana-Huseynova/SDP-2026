import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# ── Ensure project root is in path ────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

# ── Imports ──────────────────────────────────────────────────
from models.rams.model import RAMSModel
from data.probav import ProbaVDataset   


def main():
    # ── Dataset path  ─────────────────────
    base_path = "/Users/royana/Desktop/probav_data/train"

    # ── Load dataset ───────────────────────────────────────────
    dataset = ProbaVDataset(
        base_path=base_path,
        channel="NIR",   # or "RED"
        max_t=9
    )

    print(f"Total scenes: {len(dataset)}")

    # ── Get one sample ─────────────────────────────────────────
    lr, hr, mask, name = dataset[0]

    print(f"\nScene: {name}")
    print(f"LR shape: {lr.shape}")
    print(f"HR shape: {hr.shape}")

    # ── Load model ─────────────────────────────────────────────
    model = RAMSModel(band="NIR")

    # ── Run inference ──────────────────────────────────────────
    sr = model.predict(lr)

    print(f"SR shape: {sr.shape}")

    # ── Prepare images ─────────────────────────────────────────
    lr_mean = lr.mean(axis=0)

    # ── Plot ───────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))

    # LR
    plt.subplot(1, 3, 1)
    plt.title("LR (mean of frames)")
    plt.imshow(lr_mean, cmap='gray')
    plt.axis('off')

    # SR
    plt.subplot(1, 3, 2)
    plt.title("SR (RAMS output)")
    plt.imshow(sr, cmap='gray')
    plt.axis('off')

    # HR
    plt.subplot(1, 3, 3)
    plt.title("HR (ground truth)")
    plt.imshow(hr, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()