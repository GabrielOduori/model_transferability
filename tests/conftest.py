"""
Pytest configuration and shared fixtures for model transferability tests.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Dict
import gpytorch


@pytest.fixture
def device():
    """Fixture for computing device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Fixture for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def simple_1d_data(random_seed) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple 1D regression data for basic tests.

    Returns:
        (train_x, train_y): Training data tensors
    """
    n_samples = 50
    x = torch.linspace(0, 1, n_samples).unsqueeze(-1)
    y = torch.sin(2 * np.pi * x).squeeze() + 0.1 * torch.randn(n_samples)
    return x, y


@pytest.fixture
def simple_2d_data(random_seed) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple 2D regression data for testing.

    Returns:
        (train_x, train_y): Training data tensors
    """
    n_samples = 100
    x1 = torch.rand(n_samples)
    x2 = torch.rand(n_samples)
    x = torch.stack([x1, x2], dim=1)
    y = torch.sin(3 * x1) + torch.cos(4 * x2) + 0.1 * torch.randn(n_samples)
    return x, y


@pytest.fixture
def spatiotemporal_data(random_seed) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthetic spatio-temporal data for testing.

    Returns:
        (train_x, train_y): Training data with [spatial_dim1, spatial_dim2, temporal_dim]
    """
    n_samples = 100

    # Spatial coordinates (lat, lon)
    spatial = torch.rand(n_samples, 2)

    # Temporal coordinate (time)
    temporal = torch.rand(n_samples, 1)

    # Combine features
    x = torch.cat([spatial, temporal], dim=1)

    # Generate y with spatial and temporal patterns
    y = (torch.sin(2 * np.pi * spatial[:, 0]) *
         torch.cos(2 * np.pi * spatial[:, 1]) *
         torch.exp(-temporal.squeeze()) +
         0.1 * torch.randn(n_samples))

    return x, y


@pytest.fixture
def source_target_1d(random_seed) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Source and target domain data for transfer learning (1D).

    Returns:
        Dictionary with 'source' and 'target' keys, each containing (x, y) tuples
    """
    # Source domain: x in [0, 0.6]
    n_source = 80
    x_source = torch.rand(n_source).unsqueeze(-1) * 0.6
    y_source = torch.sin(2 * np.pi * x_source).squeeze() + 0.1 * torch.randn(n_source)

    # Target domain: x in [0.4, 1.0] (partial overlap)
    n_target = 30
    x_target = (0.4 + torch.rand(n_target) * 0.6).unsqueeze(-1)
    # Similar function but with slight shift
    y_target = torch.sin(2 * np.pi * x_target + 0.2).squeeze() + 0.1 * torch.randn(n_target)

    return {
        'source': (x_source, y_source),
        'target': (x_target, y_target)
    }


@pytest.fixture
def source_target_2d(random_seed) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Source and target domain data for transfer learning (2D).

    Returns:
        Dictionary with 'source' and 'target' keys
    """
    # Source domain: uniform distribution
    n_source = 150
    x_source = torch.rand(n_source, 2)
    y_source = torch.sin(3 * x_source[:, 0]) + torch.cos(4 * x_source[:, 1]) + 0.1 * torch.randn(n_source)

    # Target domain: shifted distribution
    n_target = 50
    x_target = 0.3 + torch.rand(n_target, 2) * 0.7
    y_target = torch.sin(3 * x_target[:, 0] + 0.5) + torch.cos(4 * x_target[:, 1]) + 0.15 * torch.randn(n_target)

    return {
        'source': (x_source, y_source),
        'target': (x_target, y_target)
    }


@pytest.fixture
def feature_mismatch_data(random_seed) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Source and target data with different feature dimensions (for DPTR tests).

    Returns:
        Dictionary with 'source' (3D features) and 'target' (5D features)
    """
    n_source = 100
    n_target = 50

    # Source: 3 features
    x_source = torch.randn(n_source, 3)
    y_source = x_source[:, 0] + 2 * x_source[:, 1] - x_source[:, 2] + 0.1 * torch.randn(n_source)

    # Target: 5 features (related transformation)
    x_target = torch.randn(n_target, 5)
    # First 3 dims have similar relationship, last 2 are noise
    y_target = x_target[:, 0] + 2 * x_target[:, 1] - x_target[:, 2] + 0.1 * torch.randn(n_target)

    return {
        'source': (x_source, y_source),
        'target': (x_target, y_target)
    }


@pytest.fixture
def trained_likelihood(simple_1d_data):
    """
    Pre-trained likelihood for quick model initialization.

    Returns:
        GaussianLikelihood instance
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # Set reasonable noise level
    likelihood.noise = torch.tensor(0.1)
    return likelihood


@pytest.fixture
def sample_predictions(random_seed) -> Dict[str, np.ndarray]:
    """
    Sample predictions and ground truth for metrics testing.

    Returns:
        Dictionary with 'y_true', 'y_pred', 'y_std'
    """
    n_samples = 100

    # Ground truth
    y_true = np.random.randn(n_samples)

    # Predictions with some error
    y_pred = y_true + 0.2 * np.random.randn(n_samples)

    # Uncertainty estimates (well-calibrated)
    y_std = np.abs(0.2 + 0.1 * np.random.randn(n_samples))

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_std': y_std
    }


@pytest.fixture
def sample_distributions(random_seed) -> Dict[str, np.ndarray]:
    """
    Sample distributions for KL divergence testing.

    Returns:
        Dictionary with 'dist1' and 'dist2' samples
    """
    n_samples = 1000

    # Distribution 1: N(0, 1)
    dist1 = np.random.randn(n_samples)

    # Distribution 2: N(0.5, 1.2^2)
    dist2 = 0.5 + 1.2 * np.random.randn(n_samples)

    return {
        'dist1': dist1,
        'dist2': dist2
    }


@pytest.fixture
def mock_trained_model(simple_1d_data, trained_likelihood, device):
    """
    Mock trained GP model for transfer learning tests.

    Returns:
        Tuple of (model, likelihood) ready for transfer
    """
    from src.models.gp_model import BaselineGP

    train_x, train_y = simple_1d_data
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    model = BaselineGP(train_x, train_y, trained_likelihood, ard=False)
    model = model.to(device)

    # Set reasonable hyperparameters (simulate training)
    model.covar_module.outputscale = torch.tensor(1.0, device=device)
    model.covar_module.base_kernel.lengthscale = torch.tensor([[0.2]], device=device)

    return model, trained_likelihood


@pytest.fixture
def test_train_split(simple_1d_data):
    """
    Split data into train and test sets.

    Returns:
        Dictionary with 'train' and 'test' keys
    """
    x, y = simple_1d_data
    n_total = len(x)
    n_train = int(0.7 * n_total)

    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return {
        'train': (x[train_idx], y[train_idx]),
        'test': (x[test_idx], y[test_idx])
    }


# Helper functions for tests

def assert_tensor_finite(tensor: torch.Tensor, name: str = "tensor"):
    """Assert that tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf values"


def assert_positive_definite(matrix: torch.Tensor, name: str = "matrix"):
    """Assert that matrix is positive definite."""
    try:
        torch.linalg.cholesky(matrix)
    except RuntimeError:
        pytest.fail(f"{name} is not positive definite")


def train_gp_quick(model, likelihood, train_x, train_y, n_iter: int = 50):
    """
    Quick training helper for tests.

    Args:
        model: GP model
        likelihood: Likelihood
        train_x: Training inputs
        train_y: Training targets
        n_iter: Number of iterations

    Returns:
        Final loss value
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return loss.item()


# Mark for pytest-cov
pytest_plugins = []
