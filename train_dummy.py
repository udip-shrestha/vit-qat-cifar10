import torch
from transformers import AutoModelForImageClassification
from qat_config import quant_config
from quantize_utils import prepare_qat_model

# setup 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name)
model = prepare_qat_model(model, quant_config).to(device)

# dummy data (batch of 8, 3×224×224)
inputs = torch.randn(8, 3, 224, 224).to(device)
labels = torch.randint(0, 1000, (8,)).to(device)

# optimizer + loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

#  training sanity loop 
model.train()
for step in range(3):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Step {step+1}: loss={loss.item():.4f}")

print("Dummy QAT training ran successfully ")
