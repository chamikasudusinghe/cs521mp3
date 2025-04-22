import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from resnet_torch import ResNet18
import copy

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

    base_model = ResNet18()
    base_model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location="cpu"))
    base_model.eval()
    baseline_acc = evaluate(base_model, testloader)
    print(f"baseline accuracy: {baseline_acc:.2f}%")

    low = 0
    high = 100
    best_k = 0

    while low <= high:
        mid = (low + high) // 2
        model = copy.deepcopy(base_model)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=mid / 100.0)
        acc = evaluate(model, testloader)
        print(f"prune {mid}% -> accuracy: {acc:.2f}%")

        if acc >= baseline_acc - 2.0:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    print(f"maximum pruning % with ≤2% drop: {best_k}%")