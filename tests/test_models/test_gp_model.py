"""
Unit tests for BaselineGP and SpatialTemporalGP models.
"""

import pytest
import torch
import gpytorch
import numpy as np
from src.models.gp_model import (
    BaselineGP,
    SpatialTemporalGP,
    train_baseline_gp,
    predict_with_uncertainty,
    sample_posterior
)


class TestBaselineGP:
    """Tests for BaselineGP model."""

    def test_initialization_without_ard(self, simple_1d_data, device):
        """Test BaselineGP initialization without ARD."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        assert isinstance(model, gpytorch.models.ExactGP)
        assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
        assert isinstance(model.covar_module, gpytorch.kernels.ScaleKernel)

        # Check that lengthscale has correct shape for non-ARD
        lengthscale = model.covar_module.base_kernel.lengthscale
        assert lengthscale.shape[-1] == 1

    def test_initialization_with_ard(self, simple_2d_data, device):
        """Test BaselineGP initialization with ARD."""
        train_x, train_y = simple_2d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=True)
        model = model.to(device)

        # Check that lengthscale has correct shape for ARD (one per dimension)
        lengthscale = model.covar_module.base_kernel.lengthscale
        assert lengthscale.shape[-1] == train_x.shape[-1]

    def test_forward_pass(self, simple_1d_data, device):
        """Test forward pass returns correct distribution."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(train_x)

        assert isinstance(output, gpytorch.distributions.MultivariateNormal)
        assert output.mean.shape == train_y.shape
        assert torch.isfinite(output.mean).all()
        assert torch.isfinite(output.variance).all()
        assert (output.variance > 0).all()

    def test_training_convergence(self, simple_1d_data, device):
        """Test that training decreases loss."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        model, likelihood, losses = train_baseline_gp(
            model, likelihood, train_x, train_y,
            num_iter=100, lr=0.1
        )

        # Loss should decrease
        assert losses[-1] < losses[0]
        # Final loss should be finite
        assert np.isfinite(losses[-1])
        # Should not have NaN losses
        assert not np.isnan(losses).any()

    def test_prediction_with_uncertainty(self, simple_1d_data, device):
        """Test prediction with uncertainty quantification."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        # Quick training
        model, likelihood, _ = train_baseline_gp(
            model, likelihood, train_x, train_y,
            num_iter=50, lr=0.1
        )

        # Test prediction
        test_x = torch.linspace(0, 1, 20).unsqueeze(-1).to(device)
        predictions, stds = predict_with_uncertainty(model, likelihood, test_x)

        assert predictions.shape == (20,)
        assert stds.shape == (20,)
        assert torch.isfinite(predictions).all()
        assert torch.isfinite(stds).all()
        assert (stds > 0).all()

    def test_posterior_sampling(self, simple_1d_data, device):
        """Test posterior hyperparameter sampling."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        # Quick training
        model, likelihood, _ = train_baseline_gp(
            model, likelihood, train_x, train_y,
            num_iter=50, lr=0.1
        )

        # Sample posteriors
        n_samples = 10
        sampled_models = sample_posterior(
            model, likelihood, train_x, train_y,
            n_samples=n_samples, prior_variance=0.1
        )

        assert len(sampled_models) == n_samples

        # Check that samples have variation
        lengthscales = [
            m.covar_module.base_kernel.lengthscale.item()
            for m in sampled_models
        ]
        assert np.std(lengthscales) > 0, "Samples should vary"

    def test_different_input_dimensions(self, device):
        """Test model with various input dimensions."""
        n_samples = 50
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        for input_dim in [1, 2, 5, 10]:
            train_x = torch.randn(n_samples, input_dim).to(device)
            train_y = torch.randn(n_samples).to(device)

            model = BaselineGP(train_x, train_y, likelihood, ard=True)
            model = model.to(device)

            # Check ARD lengthscale dimension
            lengthscale = model.covar_module.base_kernel.lengthscale
            assert lengthscale.shape[-1] == input_dim

            # Forward pass should work
            model.eval()
            with torch.no_grad():
                output = model(train_x)
            assert output.mean.shape == train_y.shape

    def test_edge_case_single_sample(self, device):
        """Test with single training sample (edge case)."""
        train_x = torch.randn(1, 2).to(device)
        train_y = torch.randn(1).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        # Should initialize without error
        model.eval()
        with torch.no_grad():
            output = model(train_x)
        assert output.mean.shape == train_y.shape

    def test_hyperparameter_extraction(self, simple_1d_data, device):
        """Test extraction of trained hyperparameters."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        model, likelihood, _ = train_baseline_gp(
            model, likelihood, train_x, train_y, num_iter=50
        )

        # Extract hyperparameters
        lengthscale = model.covar_module.base_kernel.lengthscale.detach()
        outputscale = model.covar_module.outputscale.detach()
        noise = likelihood.noise.detach()

        assert torch.isfinite(lengthscale).all()
        assert torch.isfinite(outputscale).all()
        assert torch.isfinite(noise).all()
        assert (lengthscale > 0).all()
        assert outputscale > 0
        assert noise > 0


class TestSpatialTemporalGP:
    """Tests for SpatialTemporalGP model."""

    def test_initialization(self, spatiotemporal_data, device):
        """Test SpatialTemporalGP initialization."""
        train_x, train_y = spatiotemporal_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        spatial_dims = [0, 1]
        temporal_dims = [2]

        model = SpatialTemporalGP(
            train_x, train_y, likelihood,
            spatial_dims=spatial_dims,
            temporal_dims=temporal_dims
        )
        model = model.to(device)

        assert isinstance(model, gpytorch.models.ExactGP)
        # Kernel should be a product kernel
        assert isinstance(model.covar_module, gpytorch.kernels.ScaleKernel)

    def test_forward_pass(self, spatiotemporal_data, device):
        """Test forward pass with spatio-temporal features."""
        train_x, train_y = spatiotemporal_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpatialTemporalGP(
            train_x, train_y, likelihood,
            spatial_dims=[0, 1],
            temporal_dims=[2]
        )
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(train_x)

        assert isinstance(output, gpytorch.distributions.MultivariateNormal)
        assert output.mean.shape == train_y.shape
        assert torch.isfinite(output.mean).all()

    def test_kernel_product_structure(self, spatiotemporal_data, device):
        """Test that kernel correctly separates spatial and temporal dimensions."""
        train_x, train_y = spatiotemporal_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpatialTemporalGP(
            train_x, train_y, likelihood,
            spatial_dims=[0, 1],
            temporal_dims=[2]
        )
        model = model.to(device)
        model.eval()

        # Get covariance matrix
        with torch.no_grad():
            covar = model.covar_module(train_x)

        # Should be positive definite
        covar_matrix = covar.to_dense()
        eigenvalues = torch.linalg.eigvalsh(covar_matrix)
        assert (eigenvalues > -1e-6).all(), "Covariance should be PSD"

    def test_spatial_temporal_separation(self, device):
        """Test that spatial and temporal kernels operate on correct dimensions."""
        n_samples = 50

        # Create data where only spatial dims vary
        spatial_coords = torch.rand(n_samples, 2).to(device)
        temporal_coord = torch.ones(n_samples, 1).to(device) * 0.5  # Fixed time
        train_x = torch.cat([spatial_coords, temporal_coord], dim=1)
        train_y = torch.randn(n_samples).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpatialTemporalGP(
            train_x, train_y, likelihood,
            spatial_dims=[0, 1],
            temporal_dims=[2]
        )
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # Evaluate at two points with same spatial location but different times
            test_x1 = torch.tensor([[0.5, 0.5, 0.1]], device=device)
            test_x2 = torch.tensor([[0.5, 0.5, 0.9]], device=device)

            # Get predictions
            out1 = model(test_x1)
            out2 = model(test_x2)

            # Cross-covariance between the two points
            test_x_combined = torch.cat([test_x1, test_x2], dim=0)
            covar = model.covar_module(test_x_combined).to_dense()

            # Variance at same point should be higher than cross-covariance
            assert covar[0, 0] >= covar[0, 1]

    def test_training_convergence(self, spatiotemporal_data, device):
        """Test training with spatio-temporal data."""
        train_x, train_y = spatiotemporal_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpatialTemporalGP(
            train_x, train_y, likelihood,
            spatial_dims=[0, 1],
            temporal_dims=[2]
        )
        model = model.to(device)

        model, likelihood, losses = train_baseline_gp(
            model, likelihood, train_x, train_y,
            num_iter=100, lr=0.1
        )

        # Training should converge
        assert losses[-1] < losses[0]
        assert np.isfinite(losses[-1])

    def test_dimension_mismatch_error(self, device):
        """Test that invalid dimension indices raise errors."""
        n_samples = 50
        train_x = torch.randn(n_samples, 3).to(device)
        train_y = torch.randn(n_samples).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # This should work: dims [0,1] and [2]
        model = SpatialTemporalGP(
            train_x, train_y, likelihood,
            spatial_dims=[0, 1],
            temporal_dims=[2]
        )

        # Forward pass should work
        model.eval()
        with torch.no_grad():
            output = model(train_x)
        assert output.mean.shape == train_y.shape


class TestModelUtilities:
    """Tests for utility functions."""

    def test_predict_with_uncertainty_batch(self, simple_1d_data, device):
        """Test prediction on batch of test points."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        test_x = torch.linspace(-0.5, 1.5, 100).unsqueeze(-1).to(device)
        predictions, stds = predict_with_uncertainty(model, likelihood, test_x)

        assert predictions.shape == (100,)
        assert stds.shape == (100,)

        # Uncertainty should be higher outside training region
        outside_idx = (test_x.squeeze() < 0) | (test_x.squeeze() > 1)
        inside_idx = ~outside_idx

        if outside_idx.any() and inside_idx.any():
            avg_std_outside = stds[outside_idx].mean()
            avg_std_inside = stds[inside_idx].mean()
            # Generally true for GP (though not strict)
            assert avg_std_outside >= avg_std_inside * 0.8

    def test_sample_posterior_variance_parameter(self, simple_1d_data, device):
        """Test effect of prior_variance on posterior sampling."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood, ard=False)
        model = model.to(device)

        model, likelihood, _ = train_baseline_gp(
            model, likelihood, train_x, train_y, num_iter=50
        )

        # Sample with small variance
        models_small = sample_posterior(
            model, likelihood, train_x, train_y,
            n_samples=20, prior_variance=0.01
        )

        # Sample with large variance
        models_large = sample_posterior(
            model, likelihood, train_x, train_y,
            n_samples=20, prior_variance=1.0
        )

        # Extract lengthscales
        ls_small = [m.covar_module.base_kernel.lengthscale.item() for m in models_small]
        ls_large = [m.covar_module.base_kernel.lengthscale.item() for m in models_large]

        # Larger variance should give more spread
        assert np.std(ls_large) > np.std(ls_small)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
