import torch
import torch.nn as nn
import torch.quantization as tq
from typing import Set

def apply_dynamic_int8_quantization(
    model: nn.Module, 
    layers_to_quantize: Set = {nn.Linear}
) -> nn.Module:
    """
    Applies dynamic INT8 quantization to a given model.

    This function moves the model to CPU, sets it to eval mode,
    removes weight normalization (if found), and then applies
    dynamic quantization.

    Args:
        model: The trained FP32 PyTorch model.
        layers_to_quantize: A set of layer types to quantize (e.g., {nn.Linear, nn.Conv2d}).

    Returns:
        A new, quantized INT8 model (on CPU).
    """
    
    # 1. Move model to CPU and set to evaluation mode
    # Quantization operations are typically optimized for CPU inference
    mdl = model.to("cpu").eval()

    # 2. Prepare the model for quantization.
    # WeightNorm is incompatible with quantization and must be removed first.
    # This logic is specific to your _MLP_ResBN's 'head' layer.
    try:
        if hasattr(mdl, 'head'):
            nn.utils.remove_weight_norm(mdl.head)
    except Exception as e:
        print(f"Warning: Could not remove weight norm from 'head'. {e}")
        pass
    
    # 3. Apply dynamic quantization
    # This converts the weights of specified layers to INT8
    # and inserts ops to dynamically quantize activations at runtime.
    model_int8 = tq.quantize_dynamic(
        mdl, layers_to_quantize, dtype=torch.qint8, inplace=False
    )

    return model_int8