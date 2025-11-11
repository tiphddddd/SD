import torch
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(
    X: torch.Tensor, 
    y: torch.Tensor, 
    batch_size: int, 
    shuffle: bool = True
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from input Tensors.

    Args:
        X: The input features tensor.
        y: The target labels tensor.
        batch_size: The batch size for the loader.
        shuffle: Whether to shuffle the data (default True).

    Returns:
        A PyTorch DataLoader instance.
    """
    
    # 1. Create a TensorDataset
    # This dataset wraps the features (X) and labels (y) tensors.
    ds = TensorDataset(X, y)
    
    # 2. Create the DataLoader
    # num_workers=0: Data is loaded in the main process (good for simple tensor datasets).
    # pin_memory=False: No need to pin memory when using CPU.
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=False,      # Keep the last batch even if it's smaller
        num_workers=0,        # Use 0 workers for simple TensorDataset
        pin_memory=False
    )
    
    return loader