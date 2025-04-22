import torch
import torch.nn as nn
from resnet_torch import ResNet18

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

model = ResNet18()
model.eval()

conv_layers = []
linear_layers = []

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        conv_layers.append((name, module))
    elif isinstance(module, nn.Linear):
        linear_layers.append((name, module))

print("conv layers")
for name, layer in conv_layers:
    num_params = count_parameters(layer)
    print(f"{name}: {num_params} params")

print("\nlinear layers")
for name, layer in linear_layers:
    num_params = count_parameters(layer)
    print(f"{name}: {num_params} params")

total_params = sum(count_parameters(layer) for _, layer in conv_layers + linear_layers)
print(f"\ntotal conv + linear params: {total_params}")