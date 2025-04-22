import torch
import torch.nn as nn
import torch.quantization
import torchvision
import torchvision.transforms as transforms
import os
import time
from resnet_torch import ResNet18

torch.backends.quantized.engine = 'qnnpack'

# CIFAR-10 preprocessing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# load fp32 model
model_fp32 = ResNet18()
model_fp32.load_state_dict(torch.load("resnet18_cifar10.pth", map_location="cpu"))
model_fp32.eval()

def print_size_of_model(model, filename="temp.p"):
    torch.save(model.state_dict(), filename)
    size = os.path.getsize(filename) / 1e6
    print(f"model size: {size:.2f} mb")
    os.remove(filename)
    return size

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"accuracy: {acc:.2f}%")
    return acc

def measure_inference_time(model, data_loader, device="cpu", num_batches=10):
    model.eval()
    model.to(device)
    total_time = 0.0
    with torch.inference_mode():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            start_time = time.time()
            _ = model(images)
            end_time = time.time()
            total_time += end_time - start_time
    avg_time = total_time / num_batches
    print(f"average inference time per batch ({device}): {avg_time * 1000:.2f} ms")
    return avg_time

if __name__ == "__main__":
    print("fp32 model")
    print_size_of_model(model_fp32)
    evaluate(model_fp32, testloader)

    model_to_quantize = ResNet18()
    model_to_quantize.load_state_dict(model_fp32.state_dict())
    model_to_quantize.eval()

    model_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model_to_quantize, inplace=True)

    print("calibrating...")
    evaluate(model_to_quantize, testloader)
    model_to_quantize.to('cpu')

    torch.quantization.convert(model_to_quantize, inplace=True)
    print("quantized int8 model")
    print_size_of_model(model_to_quantize)
    evaluate(model_to_quantize, testloader)

    print("\n inference time comparison")
    print("fp32 model:")
    measure_inference_time(model_fp32, testloader)

    print("quantized int8 model:")
    measure_inference_time(model_to_quantize, testloader)