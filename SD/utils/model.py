import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    """
    A Bottleneck Residual Block.
    Dimension flow: d -> b -> d (e.g., 256 -> 64 -> 256)
    """
    def __init__(self, d=256, b=64):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d)
        self.act1 = nn.SiLU(inplace=True)  # Use SiLU activation
        self.fc1 = nn.Linear(d, b, bias=False)  # Bottleneck layer (down-projection)
        self.bn2 = nn.BatchNorm1d(b)
        self.act2 = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(b, d, bias=False)  # Expansion layer (up-projection)
        # Initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, nonlinearity='relu')

    def forward(self, x):
        h = self.fc1(self.act1(self.bn1(x)))
        h = self.fc2(self.act2(self.bn2(h)))
        return x + h  # Residual connection

class _MLP_ResBN(nn.Module):
    """
    The main MLP model with a Bottleneck block and WeightNorm head.
    """
    def __init__(self, in_dim=784, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.SiLU(inplace=True)
        self.block = Bottleneck(d=256, b=64)  # The residual block
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU(inplace=True)  # Note: Uses ReLU here
        
        # Output head
        head = nn.Linear(128, out_dim, bias=True)
        self.head = nn.utils.weight_norm(head)  # Apply Weight Normalization
        
        # Initialize weights for linear layers
        for m in [self.fc1, self.fc2, head]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Flatten input if it's 3D (e.g., [batch, 28, 28])
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.block(x)
        x = self.act2(self.bn2(self.fc2(x)))
        return self.head(x)