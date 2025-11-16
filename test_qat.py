from transformers import AutoModelForImageClassification
from quantize_utils import prepare_qat_model
from qat_config import quant_config


model_name = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name)

model = prepare_qat_model(model, quant_config)
print("QAT model prepared successfully.")

for name, module in model.named_modules():
    if "FakeQuantize" in module.__class__.__name__:
        print(f"Found fake quantizer in: {name}")
