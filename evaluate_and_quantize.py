import torch
from torch import nn
from transformers import AutoModelForImageClassification
from load_cifar10 import CIFAR10Torch
from torchvision import transforms
from datasets import load_dataset
import torch.ao.quantization as tq
import os, time

hf_dataset = load_dataset("cifar10")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_dataset = CIFAR10Torch(hf_dataset["test"], transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)



model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = nn.Linear(768, 10)
model.load_state_dict(torch.load("vit_qat_cifar10_fullfinetune.pth", map_location="cpu"), strict=False)

model.eval()


correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
fp32_acc = 100 * correct / total
print(f"FP32 (QAT-trained) Model Accuracy: {fp32_acc:.2f}%")


model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
quantized_model = tq.convert(model.eval(), inplace=False)

torch.save(quantized_model.state_dict(), "vit_qat_cifar10_int8.pth")
print("Quantized INT8 model saved as vit_qat_cifar10_int8.pth")




fp32_size = os.path.getsize("vit_qat_cifar10_fullfinetune.pth") / (1024*1024)
int8_size = os.path.getsize("vit_qat_cifar10_int8.pth") / (1024*1024)
print(f"FP32 model size: {fp32_size:.2f} MB")
print(f"INT8 model size: {int8_size:.2f} MB")
print(f"Compression ratio: {fp32_size/int8_size:.2f}x smaller")



dummy_input = torch.randn(1, 3, 224, 224)

# FP32
start = time.time()
for _ in range(50):
    _ = model(dummy_input)
print("FP32 average inference:", (time.time()-start)/50, "sec")

# INT8
start = time.time()
for _ in range(50):
    _ = quantized_model(dummy_input)
print("INT8 average inference:", (time.time()-start)/50, "sec")

