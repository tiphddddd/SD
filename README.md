![Documentation Check](https://github.com/tiphddddd/SD/actions/workflows/check-docs.yml/badge.svg)
# 1 Minute Permuted MNIST
## 1.Competition Overview
This is a fast adaptation challenge where agents must quickly learn to classify MNIST digits that have been randomly permuted. The key challenge: both pixels AND labels are randomly shuffled for each task, requiring agents to adapt from scratch within strict time and resource constraints.
---
## 🚀 Quick Start

Get your project up and running quickly.
And the environment and data we used in this project is from ML_Arena 1 minute permuted mnist.
### 🌟Please don't forget to get it via https://github.com/ml-arena/permuted_mnist

1.  Clone the repository
    ```bash
    git clone [https://github.com/tiphddddd/SD.git](https://github.com/tiphddddd/SD.git)
    ```

2.  Navigate to the project directory
    ```bash
    cd SD
    ```

3.  Install package and install dependencies
    you can find dependencies in the file requirements.txt but it's not required
    ```bash
    pip install -e .
    ```
---

## ✨ Features
* **High Efficiency:** Achieves high accuracy with less computational cost.
* **Multi-Faceted Optimization:** Employs a combination of feature engineering, advanced optimizers, and model compression to ensure high efficiency.

---
## Example Usage
```python
import numpy as np
import pandas as pd
import torch
from SD.agent import Tochmlp as BestAgent
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv

# Create environment
env = PermutedMNISTEnv(number_episodes=10)
env.set_seed(42)

# Initialize your agent
agent = BestAgent(output_dim=10, seed=42, epochs=10, batch_size=128, lr=1e-3)
# Evaluation loop (this simulates the competition evaluation)
total_time = 0
accuracies = []

for episode in range(10):
    # Get next task with new permutations
    task = env.get_next_task()
    if task is None:
        break

    # Reset agent for new task
    agent.reset()

    import time
    start = time.time()

    # Train (must complete within time limit)
    agent.train(task['X_train'], task['y_train'])

    # Predict
    predictions = agent.predict(task['X_test'])

    elapsed = time.time() - start
    total_time += elapsed

    # Evaluate
    accuracy = env.evaluate(predictions, task['y_test'])
    accuracies.append(accuracy)

    print(f"Episode {episode + 1}: Accuracy: {accuracy:.3f}, Time: {elapsed:.2f}s")

print(f"\nFinal Score: {np.mean(accuracies):.3f}")
print(f"Total Time: {total_time:.2f}s")
print(f"Status: {'PASS ✅' if total_time < 600 else 'FAIL ❌ (timeout)'}")
```
---

## Report
check out report.ipynb for:
* baseline 
* optimasation : Feature Engineering, Improvement of Model Structure, Optimizer, Model Compression
* Hyperparameter Tuning



