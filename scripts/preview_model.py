import os, numpy as np, matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data.probav import ProbaVDataset
from models.deepsum.model import DeepSUMModel

ds = ProbaVDataset('/Users/royana/Desktop/probav_data/train', channel='NIR')
lr_stack, hr, masks, name = ds[0]

with DeepSUMModel() as model:
    sr = model.predict(lr_stack, masks.astype(bool))

print(f'LR  range: {lr_stack[0].min():.4f} - {lr_stack[0].max():.4f}')
print(f'SR  range: {sr.min():.4f} - {sr.max():.4f}')
print(f'HR  range: {hr.min():.4f} - {hr.max():.4f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0f1117')
for ax in axes:
    ax.set_facecolor('#0f1117')

# Auto-scale each image independently
axes[0].imshow(lr_stack[0], cmap='gray', interpolation='nearest')
axes[0].set_title('LR best frame (128x128)', color='white')
axes[0].axis('off')

axes[1].imshow(sr, cmap='gray', interpolation='nearest')
axes[1].set_title(f'DeepSUM output (384x384)', color='#5b8dee')
axes[1].axis('off')

axes[2].imshow(hr, cmap='gray', interpolation='nearest')
axes[2].set_title('HR ground truth (384x384)', color='#4caf7d')
axes[2].axis('off')

plt.suptitle(f'Scene: {name}', color='white')
plt.tight_layout()
plt.savefig('model_output_preview.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print('Saved → model_output_preview.png')
plt.show()