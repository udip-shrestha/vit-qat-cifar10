import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModelForImageClassification
from quantize_utils import prepare_qat_model
from qat_config import quant_config
from load_cifar10 import CIFAR10Torch
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# 1. Load CIFAR-10 dataset
# -----------------------------
hf_dataset = load_dataset("cifar10")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = CIFAR10Torch(hf_dataset["train"], transform=transform)
test_dataset = CIFAR10Torch(hf_dataset["test"], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0)

# -----------------------------
# 2. Load pretrained ViT model
# -----------------------------
model_name = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name)

# Replace classification head for CIFAR-10 (10 classes)
model.classifier = nn.Linear(in_features=768, out_features=10)

# Prepare full model for Quantization-Aware Training
model = prepare_qat_model(model, quant_config)
model.to(device)

# -----------------------------
# 3. Training setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 3  # start small; increase later

# -----------------------------
# 4. Training + validation loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = 100 * correct / total

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# -----------------------------
# 5. Save model
# -----------------------------
torch.save(model.state_dict(), "vit_qat_cifar10_fullfinetune.pth")
print("Model saved as vit_qat_cifar10_fullfinetune.pth")
 