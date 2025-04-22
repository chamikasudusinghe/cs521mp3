import torch
import numpy as np
from resnet_torch import ResNet18

model = ResNet18()
state_dict = torch.load("resnet18_cifar10.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
flat_params = {}
for key, value in model.state_dict().items():
    flat_key = key.replace("running_mean", "mean").replace("running_var", "var")
    flat_params[flat_key] = value.cpu().numpy()

# save as npz
np.savez("resnet18_pytorch_weights.npz", **flat_params)