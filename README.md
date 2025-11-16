
# ViT Quantization-Aware Training (QAT) on CIFAR-10

This project fine-tunes `google/vit-base-patch16-224` using Quantization-Aware Training (QAT) with PyTorch.

## Features
- Full fine-tuning on CIFAR-10
- Layer-wise quantization configuration
- INT8 conversion and evaluation
- Accuracy, model size, and inference benchmarks

## Files
- `train_cifar10_qat.py` — QAT fine-tuning script
- `evaluate_and_quantize.py` — Converts QAT to INT8 and compares results
- `quantize_utils.py` — Helper functions for quantization
- `qat_config.py` — Configurable layer-bitwidth setup
- `load_cifar10.py` — CIFAR-10 dataset loader

## Results
| Metric | FP32 | INT8 |
|--------|------|------|
| Accuracy | 95.9% | ~95.9% |
| Compression | ~1x (state_dict) |
| Inference speed | 0.13s → 0.12s |

---
**Author:** Udip Shrestha  
**Environment:** PyTorch, Transformers, TorchVision, Hugging Face Datasets

