import torch
import numpy as np

def _as_float_01(x_np: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy array to a Float Tensor scaled to [0, 1].
    """
    x = torch.from_numpy(x_np).float()
    # Normalize if data is in [0, 255] range
    if x.max() > 1.0:
        x = x / 255.0
    return x

def _labels_1d(y_np: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy label array to a 1D Long Tensor.
    """
    y = np.asarray(y_np).reshape(-1).astype(np.int64, copy=False)
    return torch.from_numpy(y)

def _l2_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Applies L2 normalization to each sample in the batch.
    This is the Feature Engineering (FE) step.
    """
    orig = x.shape
    # Flatten if 3D (e.g., [B, C, H, W] or [B, H, W])
    if x.dim() == 3:
        x = x.view(x.size(0), -1)
    
    # Calculate L2 norm per sample
    n = x.norm(dim=1, keepdim=True).clamp_min(eps)
    x = x / n
    
    # Reshape back to original if needed
    if len(orig) == 3:
        x = x.view(orig)
    return x