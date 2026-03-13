import torch
import numpy as np
from model2 import ResNet

def run_ai_inference(data_array):
    model = ResNet()
    model.eval()
    # Convert numpy input to a PyTorch tensor with batch and channel dims
    tensor = torch.from_numpy(data_array).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze().numpy()