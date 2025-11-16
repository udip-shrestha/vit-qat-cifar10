from transformers import AutoModelForImageClassification, AutoImageProcessor

model_name = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

print(model)
