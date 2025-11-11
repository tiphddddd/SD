# tests/test_agent.py

import pytest
import numpy as np
import torch
import copy # To check if weights have changed

# 导入你的 agent。
# 注意：这取决于你的 PYTHONPATH。
# 你可能需要 `from your_repo.agent import MLPAgent` 
# 或者如果 `your_repo` 是根目录，就是 `from agent import MLPAgent`
from agent import Agent as BestAgent 

# --- Fixtures (可复用的测试设置) ---

@pytest.fixture
def agent_config():
    """Defines the model dimensions for consistency."""
    return {
        "input_dim": 10,  # 10 input features
        "output_dim": 3,  # 3 possible classes (0, 1, 2)
        "hidden_dims": [16, 8], # Small hidden layers for fast tests
        "seed": 42
    }

@pytest.fixture
def mlp_agent(agent_config):
    """Creates a fresh MLPAgent instance for each test."""
    return BestAgent(**agent_config)

@pytest.fixture
def dummy_train_data(agent_config):
    """Creates small, matching training data."""
    # 20 samples, 10 features (matches input_dim)
    X_train = np.random.rand(20, agent_config["input_dim"])
    # 20 labels, classes 0, 1, or 2 (matches output_dim)
    y_train = np.random.randint(0, agent_config["output_dim"], size=20)
    return X_train, y_train

@pytest.fixture
def dummy_test_data(agent_config):
    """Creates small, matching test data."""
    # 5 samples, 10 features
    X_test = np.random.rand(5, agent_config["input_dim"])
    return X_test


# --- Unit Tests ---

def test_agent_creation(mlp_agent, agent_config):
    """Test 1: Can the agent be created correctly?"""
    assert mlp_agent is not None
    assert mlp_agent.input_dim == agent_config["input_dim"]
    assert mlp_agent.output_dim == agent_config["output_dim"]
    # Check if the PyTorch model object was actually created
    assert isinstance(mlp_agent.model, torch.nn.Module)

def test_train_runs_without_error(mlp_agent, dummy_train_data):
    """Test 2: Can the train() method be called without crashing?"""
    X_train, y_train = dummy_train_data
    try:
        # Run training for just one epoch (fast)
        mlp_agent.train(X_train, y_train, epochs=1)
    except Exception as e:
        pytest.fail(f"agent.train() raised an exception: {e}")

def test_predict_output_shape(mlp_agent, dummy_test_data):
    """Test 3: Does predict() return an array with the correct shape?"""
    X_test = dummy_test_data
    predictions = mlp_agent.predict(X_test)
    
    assert isinstance(predictions, np.ndarray)
    # 5 samples in, 5 predictions out
    assert predictions.shape == (5,) 

def test_predict_output_range(mlp_agent, dummy_test_data, agent_config):
    """Test 4: Are the predicted class labels within the expected range?"""
    X_test = dummy_test_data
    predictions = mlp_agent.predict(X_test)
    
    # All predictions should be >= 0
    assert np.all(predictions >= 0)
    # All predictions should be < output_dim (i.e., 0, 1, or 2)
    assert np.all(predictions < agent_config["output_dim"])
    # Check if they are integers
    assert np.all(predictions == predictions.astype(int))

def test_train_changes_weights(mlp_agent, dummy_train_data):
    """
    Test 5: (Advanced, but important)
    Does training *actually* change the model's weights?
    """
    X_train, y_train = dummy_train_data

    # 1. Get a deep copy of the weights *before* training
    # We check the first layer's weights
    initial_weights = copy.deepcopy(
        list(mlp_agent.model.parameters())[0].data
    )

    # 2. Run training
    mlp_agent.train(X_train, y_train, epochs=1)

    # 3. Get the weights *after* training
    trained_weights = list(mlp_agent.model.parameters())[0].data

    # 4. Assert that the weights are NOT identical
    #    (This proves backpropagation and optimization step happened)
    assert not torch.equal(initial_weights, trained_weights)