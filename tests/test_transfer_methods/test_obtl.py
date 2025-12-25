"""
Unit tests for CORRECTED OBTL (Optimal Bayesian Transfer Learning) implementation.

Tests the mathematically correct implementation using Wishart posterior on precision matrices.
"""

import pytest
import torch
import gpytorch
import numpy as np
from src.transfer_methods.obtl import (
    OBTLGaussianProcess,
    train_obtl_gp
)


class TestOBTLGaussianProcess:
    """Tests for corrected OBTLGaussianProcess class."""

    def test_initialization(self):
        """Test OBTL initialization with default parameters."""
        obtl = OBTLGaussianProcess(
            n_inducing_points=20,
            nu_0=25.0  # Must be > d+1 = 21
        )

        assert obtl.n_inducing == 20
        assert obtl.nu_0 == 25.0
        assert obtl.source_cov is None
        assert obtl.target_cov is None

    def test_initialization_warns_low_nu(self):
        """Test that low nu_0 triggers warning."""
        with pytest.warns(UserWarning, match="nu_0.*should be > d\\+1"):
            obtl = OBTLGaussianProcess(n_inducing_points=10, nu_0=5.0)

    def test_fit_source(self, source_target_1d, device):
        """Test fitting source model."""
        x_source, y_source = source_target_1d['source']
        x_source = x_source.to(device)
        y_source = y_source.to(device)

        obtl = OBTLGaussianProcess(n_inducing_points=10, nu_0=15.0)
        source_model, source_likelihood = obtl.fit_source(x_source, y_source)

        # Check source was fitted
        assert source_model is not None
        assert source_likelihood is not None
        assert obtl.source_cov is not None
        assert obtl.inducing_points is not None

    def test_inducing_point_selection(self, source_target_1d, device):
        """Test that inducing points are selected correctly."""
        x_source, y_source = source_target_1d['source']
        x_source = x_source.to(device)
        y_source = y_source.to(device)

        n_inducing = 15
        obtl = OBTLGaussianProcess(n_inducing_points=n_inducing, nu_0=20.0)
        obtl.fit_source(x_source, y_source)

        # Should have selected inducing points
        assert obtl.inducing_points is not None
        assert obtl.inducing_points.shape[0] == n_inducing
        assert obtl.inducing_points.shape[1] == x_source.shape[1]

    def test_covariance_extraction(self, source_target_1d, device):
        """Test covariance matrix extraction at inducing points."""
        x_source, y_source = source_target_1d['source']
        x_source = x_source.to(device)
        y_source = y_source.to(device)

        n_inducing = 10
        obtl = OBTLGaussianProcess(n_inducing_points=n_inducing, nu_0=15.0)
        obtl.fit_source(x_source, y_source)

        # Extract covariance
        assert obtl.source_cov is not None
        assert obtl.source_cov.shape == (n_inducing, n_inducing)

        # Should be positive definite
        eigenvalues = torch.linalg.eigvalsh(obtl.source_cov)
        assert (eigenvalues > -1e-6).all(), "Covariance should be PSD"

    def test_transfer_to_target(self, source_target_1d, device):
        """Test transfer to target domain."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        obtl = OBTLGaussianProcess(n_inducing_points=10, nu_0=15.0)

        # Fit source
        obtl.fit_source(x_source, y_source)

        # Transfer to target
        transferred_cov, info = obtl.transfer_to_target(
            x_target, y_target, delta=0.5
        )

        # Check outputs
        assert transferred_cov is not None
        assert transferred_cov.shape == (10, 10)
        assert isinstance(info, dict)
        assert 'weight_source' in info
        assert 'weight_target' in info
        assert 'effective_nu' in info

    def test_delta_parameter_effect(self, source_target_1d, device):
        """Test effect of delta parameter on transfer."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        obtl = OBTLGaussianProcess(n_inducing_points=10, nu_0=15.0)
        obtl.fit_source(x_source, y_source)

        # Test with delta=0 (no transfer)
        cov_0, info_0 = obtl.transfer_to_target(x_target, y_target, delta=0.0)
        assert info_0['weight_source'] == 0.0

        # Test with delta=1 (full transfer)
        cov_1, info_1 = obtl.transfer_to_target(x_target, y_target, delta=1.0)
        assert info_1['weight_source'] > info_0['weight_source']

    def test_wishart_posterior_formula(self, source_target_1d, device):
        """Test that implementation matches Wishart posterior formula."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        n_inducing = 10
        nu_0 = 15.0
        delta = 1.0

        obtl = OBTLGaussianProcess(n_inducing_points=n_inducing, nu_0=nu_0)
        obtl.fit_source(x_source, y_source)
        transferred_cov, info = obtl.transfer_to_target(x_target, y_target, delta=delta)

        # Manually compute using the formula
        n_t = x_target.shape[0]
        d = n_inducing

        Lambda_0 = torch.linalg.inv(obtl.source_cov)
        S_target_inv = torch.linalg.inv(obtl.target_cov)
        Lambda_n = delta * nu_0 * Lambda_0 + n_t * S_target_inv
        normalizer = delta * nu_0 + n_t - d - 1

        expected_cov = torch.linalg.inv(Lambda_n) / normalizer
        expected_cov = expected_cov + 1e-6 * torch.eye(d, device=device)

        # Compare
        diff = torch.norm(transferred_cov - expected_cov, p='fro')
        relative_diff = diff / torch.norm(expected_cov, p='fro')

        assert relative_diff < 0.01, "Implementation should match Wishart formula"

    def test_precision_matrix_usage(self, source_target_1d, device):
        """Verify that precision matrices are used, not simple covariance averaging."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        obtl = OBTLGaussianProcess(n_inducing_points=10, nu_0=15.0)
        obtl.fit_source(x_source, y_source)
        transferred_cov, info = obtl.transfer_to_target(x_target, y_target, delta=1.0)

        # Compute what a WRONG simple averaging would give
        weight_s = info['weight_source']
        weight_t = info['weight_target']
        wrong_cov = weight_s * obtl.source_cov + weight_t * obtl.target_cov

        # Should be DIFFERENT from simple averaging
        diff = torch.norm(transferred_cov - wrong_cov, p='fro')
        assert diff > 0.01, "Should NOT be simple covariance averaging!"

    def test_different_input_dimensions(self, device):
        """Test OBTL with different input dimensions."""
        n_source = 100
        n_target = 50

        for input_dim in [1, 2, 5]:
            x_source = torch.randn(n_source, input_dim).to(device)
            y_source = torch.randn(n_source).to(device)

            x_target = torch.randn(n_target, input_dim).to(device)
            y_target = torch.randn(n_target).to(device)

            obtl = OBTLGaussianProcess(n_inducing_points=10, nu_0=15.0)
            obtl.fit_source(x_source, y_source)
            transferred_cov, info = obtl.transfer_to_target(x_target, y_target)

            assert obtl.inducing_points.shape[1] == input_dim
            assert transferred_cov.shape[0] == 10


class TestTrainOBTLGP:
    """Test the end-to-end train_obtl_gp function."""

    def test_full_obtl_pipeline(self, source_target_2d, device):
        """Test complete OBTL training pipeline."""
        x_source, y_source = source_target_2d['source']
        x_target, y_target = source_target_2d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Train with OBTL
        model, likelihood, obtl_info = train_obtl_gp(
            x_source, y_source,
            x_target, y_target,
            n_inducing=15,
            nu_0=20.0,
            delta=0.7,
            n_iterations=50
        )

        # Check outputs
        assert model is not None
        assert likelihood is not None
        assert 'transferred_cov' in obtl_info
        assert 'weight_source' in obtl_info
        assert 'weight_target' in obtl_info

        # Make predictions
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            test_output = model(x_target)
            predictions = likelihood(test_output).mean

        assert predictions.shape == y_target.shape
        assert torch.isfinite(predictions).all()

    def test_nu_0_parameter_effect(self, source_target_1d, device):
        """Test effect of nu_0 (prior strength) parameter."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Low nu_0 (weak prior)
        _, _, info_low = train_obtl_gp(
            x_source, y_source, x_target, y_target,
            n_inducing=10, nu_0=12.0, delta=1.0, n_iterations=50
        )

        # High nu_0 (strong prior)
        _, _, info_high = train_obtl_gp(
            x_source, y_source, x_target, y_target,
            n_inducing=10, nu_0=30.0, delta=1.0, n_iterations=50
        )

        # Higher nu_0 should give more weight to source
        assert info_high['weight_source'] > info_low['weight_source']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
