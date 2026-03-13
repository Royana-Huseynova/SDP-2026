import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from probavisualizer import save_four_panel
import torch
from model2 import ResNet

# --- PATH CONFIGURATION ---
dataset_path = "/Users/royana/Desktop/probav_data" 
save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_baseline")
os.makedirs(save_folder, exist_ok=True)

band = "RED"
scene_folder = "imgset0000"
scene_path = os.path.join(dataset_path, "train", band, scene_folder)

# 1. Load files
lr_files = sorted([f for f in os.listdir(scene_path) if f.startswith("LR")])
sm_files = sorted([f for f in os.listdir(scene_path) if f.startswith("SM")])

# 2. Stack with Lighter Masking
lr_stack = []
for lr_file, sm_file in zip(lr_files, sm_files):
    lr = imageio.imread(os.path.join(scene_path, lr_file)).astype(float)
    sm = imageio.imread(os.path.join(scene_path, sm_file))
    sm_resized = np.array(Image.fromarray(sm).resize((lr.shape[1], lr.shape[0]), Image.NEAREST))
    lr[sm_resized > 5] = np.nan
    lr_stack.append(lr)

# 3. Calculate Median
lr_median = np.nanmedian(np.array(lr_stack), axis=0)
lr_median = np.nan_to_num(lr_median)

# 4. Manual Scaling
img_min, img_max = np.min(lr_median), np.max(lr_median)
viz_median = ((lr_median - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)

# 5. Save 
save_four_panel(
    imageio.imread(os.path.join(scene_path, "HR.png")), 
    viz_median, 
    np.array(Image.fromarray(viz_median).resize((384, 384), Image.BICUBIC)), 
    imageio.imread(os.path.join(scene_path, sm_files[0])), 
    band, "Median_Aggregate", [0], save_folder
)


print("Running AI Inference...")
model = ResNet()
model.eval()

# Convert median to tensor [1, 1, H, W]
input_tensor = torch.from_numpy(lr_median).float().unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    sr_output = model(input_tensor)
    ai_result = sr_output.squeeze().numpy()

# Save result to file, not the visualizer
np.save(os.path.join(save_folder, "ai_result.npy"), ai_result)
print(f"AI result saved to {save_folder}/ai_result.npy")