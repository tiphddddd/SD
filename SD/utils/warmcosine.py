import numpy as np
from torch.optim.optimizer import Optimizer

class WarmupCosine:
    """
    A Learning Rate Scheduler that combines Linear Warmup and Cosine Annealing.
    """
    def __init__(self, optimizer: Optimizer, total_epochs: int, warmup_epochs: int = 1):
        self.opt = optimizer
        self.total = total_epochs
        self.warm = warmup_epochs
        # Store the base learning rates for each param group
        self.base = [g['lr'] for g in optimizer.param_groups]
        self.ep = 0  # Current epoch counter

    def step(self):
        """Call this after each epoch."""
        self.ep += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base[i]
            if self.ep <= self.warm:
                # Linear warmup phase
                lr = base * self.ep / max(1, self.warm)
            else:
                # Cosine annealing phase
                prog = (self.ep - self.warm) / max(1, (self.total - self.warm))
                lr = 0.5 * base * (1 + np.cos(np.pi * prog))
            g['lr'] = lr