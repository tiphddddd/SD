import torch
import torch.nn as nn

class SmoothCE(nn.Module):
    """
    Implements Label Smoothing Cross Entropy Loss.
    """
    def __init__(self, eps=0.05, num_classes=10):
        super().__init__()
        self.eps = eps  # The smoothing factor
        self.C = num_classes  # Number of classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, target):
        logp = self.logsoftmax(logits)
        with torch.no_grad():
            # Create the one-hot target tensor
            t = torch.zeros_like(logp).scatter_(1, target.view(-1, 1), 1.0)
            # Apply smoothing: (1-eps) for the true class, eps/C for others
            t = t * (1 - self.eps) + self.eps / self.C
        # Calculate the mean loss (negative log-likelihood)
        return (-t * logp).sum(dim=1).mean()

# You could add other custom losses here, like FocalLoss, etc.