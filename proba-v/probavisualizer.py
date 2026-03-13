import matplotlib.pyplot as plt
import os

def save_four_panel(hr_norm, lr_norm, lr_resized_norm, qm, band, lr_file, unique_vals, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    
    # 1. HR
    axes[0].imshow(hr_norm, cmap="gray")
    axes[0].set_title(f"HR (Ground Truth)\n{band}")
    axes[0].axis("off")
    
    # 2. LR Original
    axes[1].imshow(lr_norm, cmap="gray")
    axes[1].set_title(f"LR Original\n{lr_file}")
    axes[1].axis("off")
    
    # 3. LR Upsampled
    axes[2].imshow(lr_resized_norm, cmap="gray")
    axes[2].set_title(f"LR Upsampled\n(Bicubic 3x)")
    axes[2].axis("off")
    
    # 4. Quality Mask 
    axes[3].imshow(qm, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Quality Mask\nValues: {unique_vals}")
    axes[3].axis("off")
    
    plt.tight_layout()
    
    # Save using band name and filename to avoid overwriting
    save_path = os.path.join(save_dir, f"{band}_{lr_file.replace('.png', '')}.png")
    plt.savefig(save_path)
    plt.close(fig)