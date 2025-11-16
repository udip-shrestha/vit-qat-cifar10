import torch
import torch.nn as nn
import torch.ao.quantization as tq

# Recursively apply per-layer quantization config
def apply_qat_per_layer(model, quant_config):
    qconfig = tq.get_default_qat_qconfig('fbgemm')  # fake quantization setup

    for name, module in model.named_children():
        if name in quant_config:
            print(f"Applying QAT to layer: {name} with {quant_config[name]}-bit")
            module.qconfig = qconfig
        else:
            apply_qat_per_layer(module, quant_config)  # recursive descent

    return model

# Utility to prepare the model for QAT
def prepare_qat_model(model, quant_config):
    model.train()  
    model = apply_qat_per_layer(model, quant_config)
    tq.prepare_qat(model, inplace=True)
    return model

