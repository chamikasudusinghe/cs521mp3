import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from resnet_torch import ResNet18
import copy
import csv

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

if __name__ == "__main__":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    model_fp32 = ResNet18()
    model_fp32.load_state_dict(torch.load("resnet18_cifar10.pth", map_location="cpu"))
    model_fp32.eval()

    prunable_layers = [(name, m) for name, m in model_fp32.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    results = []
    for name, layer in prunable_layers:
        model_copy = copy.deepcopy(model_fp32)
        submodule = dict(model_copy.named_modules())[name]
        try:
            prune.l1_unstructured(submodule, name='weight', amount=0.9)
            acc = evaluate(model_copy, testloader)
            results.append((name, acc))
            print(f"pruned {name}: accuracy = {acc:.2f}%")
        except Exception as e:
            print(f"skipping {name} due to error: {e}")

    with open("prune_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "acc after 90% prune"])
        writer.writerows(results)