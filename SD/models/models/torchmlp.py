"""
PyTorch MLP Agent for Permuted MNIST
Multi-layer perceptron with batch normalization
"""
import torch
from torch import nn
import numpy as np
from time import time


class Model(nn.Module):

    def __init__(self, hidden_sizes): 
        super(Model, self).__init__()

      
        layers = []
        d_in = 28 ** 2
        for i, n in enumerate(hidden_sizes):
            layers.append(nn.Linear(d_in, n))
            layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            d_in = n

        layers += [nn.Linear(d_in, 10)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)


class Agent:
    """PyTorch MLP agent for MNIST classification"""

    
    def __init__(self, output_dim: int = 10, seed: int = None,
                 hidden_sizes: list = [400, 400],
                 n_epochs: int = 10,
                 batch_size: int = 16):
        """
        Initialize the MLP agent

        Args:
            output_dim: Number of output classes (10 for MNIST digits)
            seed: Random seed for reproducibility
            hidden_sizes: List of integers for hidden layer sizes 
            n_epochs: Number of epochs to train for             
            batch_size: Batch size for training                 
        """
        self.output_dim = output_dim
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.hidden_sizes = hidden_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = Model(hidden_sizes=self.hidden_sizes)

        self.validation_fraction = 0.2
        self.verbose = True

    def reset(self):
        """Reset the agent for a new task/simulation"""
        self.model = Model(hidden_sizes=self.hidden_sizes)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the agent on the provided data

        Args:
            X_train: Training images (N, 28, 28) or (N, 784)
            y_train: Training labels (N, 1) or (N,)
        """
        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()

        N_val = int(X_train.shape[0] * self.validation_fraction)
        X_train_sub, X_val = X_train[N_val:], X_train[:N_val]
        y_train_sub, y_val = y_train[N_val:], y_train[:N_val]

        X_train_sub = torch.from_numpy(X_train_sub).float() / 255.0
        X_val = torch.from_numpy(X_val).float() / 255.0
        y_train_sub = torch.from_numpy(y_train_sub).long()
        y_val = torch.from_numpy(y_val).long()

        N = len(X_train_sub)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        for i_epoch in range(self.n_epochs): 
            perm = np.random.permutation(N)
            X = X_train_sub[perm]
            Y = y_train_sub[perm]

            for i in range(0, N, self.batch_size):
                x = X[i:i + self.batch_size]
                y = Y[i:i + self.batch_size]

                optimizer.zero_grad()
                logits = self.model(x)
                loss = ce(logits, y)
                loss.backward()
                optimizer.step()

            if self.verbose and self.validation_fraction > 0:
                y_predict = self.predict(X_val.numpy() * 255.0)
                is_correct = y_predict == y_val.numpy()
                acc = np.mean(is_correct)
                print(f"epoch {i_epoch+1}/{self.n_epochs}: {acc:0.04f}%")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data

        Args:
            X_test: Test images (N, 28, 28) or (N, 784)

        Returns:
            Class labels (N,)
        """
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float() / 255.0
        with torch.no_grad():
            logits = self.model.forward(X_test)
        return logits.argmax(-1).detach().cpu().numpy()


if __name__ == "__main__":
    
    agent = Agent(
        seed=42,
        hidden_sizes=[512, 512, 512], 
        n_epochs=15,               
        batch_size=32
    )
    

    X_train = np.random.rand(1000, 784) * 255
    y_train = np.random.randint(0, 10, (1000,))
    X_test = np.random.rand(100, 784) * 255
    y_test = np.random.randint(0, 10, (100,))


    t0 = time()
    agent.train(X_train, y_train)
    y_predict = agent.predict(X_test)
    is_correct = y_predict == y_test
    acc = np.mean(is_correct)
    print(f"Test accuracy: {acc:0.04f} in {time() - t0:.2f} seconds")