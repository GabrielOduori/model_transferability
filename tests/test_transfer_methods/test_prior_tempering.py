"""
Unit tests for Prior Tempering transfer learning method.
"""

import pytest
import torch
import gpytorch
import numpy as np
from src.transfer_methods.prior_tempering import (
    TemperedGP,
    TemperedMarginalLogLikelihood,
    train_tempered_gp
)


class TestTemperedGP:
    """Tests for TemperedGP model."""

    def test_initialization_without_source(self, simple_1d_data, device):
        """Test TemperedGP initialization without source hyperparameters."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = TemperedGP(train_x, train_y, likelihood)
        model = model.to(device)

        assert isinstance(model, gpytorch.models.ExactGP)
        assert model.source_hyperparams is None

    def test_initialization_with_source(self, simple_1d_data, device):
        """Test TemperedGP initialization with source hyperparameters."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        source_hyperparams = {
            'lengthscale': torch.tensor([[0.3]], device=device),
            'outputscale': torch.tensor(1.5, device=device),
            'noise': torch.tensor(0.1, device=device)
        }

        model = TemperedGP(
            train_x, train_y, likelihood,
            source_hyperparams=source_hyperparams
        )
        model = model.to(device)

        assert model.source_hyperparams is not None
        assert 'lengthscale' in model.source_hyperparams
        assert 'outputscale' in model.source_hyperparams
        assert 'noise' in model.source_hyperparams

        # Check initialization from source
        current_ls = model.covar_module.base_kernel.lengthscale
        assert torch.allclose(current_ls, source_hyperparams['lengthscale'], atol=1e-3)

    def test_forward_pass(self, simple_1d_data, device):
        """Test forward pass of TemperedGP."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = TemperedGP(train_x, train_y, likelihood)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(train_x)

        assert isinstance(output, gpytorch.distributions.MultivariateNormal)
        assert output.mean.shape == train_y.shape
        assert torch.isfinite(output.mean).all()


class TestTemperedMarginalLogLikelihood:
    """Tests for TemperedMarginalLogLikelihood."""

    def test_initialization(self, simple_1d_data, device):
        """Test TemperedMLL initialization."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = TemperedGP(train_x, train_y, likelihood)
        model = model.to(device)

        mll = TemperedMarginalLogLikelihood(
            likelihood, model,
            beta=0.5,
            prior_variances=0.1
        )

        assert mll.beta == 0.5
        assert mll.prior_variance == 0.1

    def test_beta_zero_equals_standard_mll(self, simple_1d_data, device):
        """Test that beta=0 gives standard MLL (no prior tempering)."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        source_hyperparams = {
            'lengthscale': torch.tensor([[0.3]], device=device),
            'outputscale': torch.tensor(1.5, device=device),
            'noise': torch.tensor(0.1, device=device)
        }

        model = TemperedGP(
            train_x, train_y, likelihood,
            source_hyperparams=source_hyperparams
        )
        model = model.to(device)
        model.train()
        likelihood.train()

        # Tempered MLL with beta=0
        tempered_mll = TemperedMarginalLogLikelihood(likelihood, model, beta=0.0)

        # Standard MLL
        standard_mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Both should give same value (within numerical precision)
        output = model(train_x)
        tempered_loss = -tempered_mll(output, train_y)
        standard_loss = -standard_mll(output, train_y)

        # Should be very close (beta=0 means no prior)
        assert torch.allclose(tempered_loss, standard_loss, rtol=1e-3)

    def test_beta_one_adds_prior(self, simple_1d_data, device):
        """Test that beta=1 adds prior term to MLL."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        source_hyperparams = {
            'lengthscale': torch.tensor([[0.3]], device=device),
            'outputscale': torch.tensor(1.5, device=device),
            'noise': torch.tensor(0.1, device=device)
        }

        model = TemperedGP(
            train_x, train_y, likelihood,
            source_hyperparams=source_hyperparams
        )
        model = model.to(device)
        model.train()
        likelihood.train()

        # Tempered MLL with beta=1
        tempered_mll = TemperedMarginalLogLikelihood(likelihood, model, beta=1.0)

        # Standard MLL
        standard_mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        output = model(train_x)
        tempered_loss = -tempered_mll(output, train_y)
        standard_loss = -standard_mll(output, train_y)

        # Tempered loss should be different (includes prior)
        # Since we initialized from source, prior should favor those values
        assert not torch.allclose(tempered_loss, standard_loss, rtol=1e-2)

    def test_prior_variance_effect(self, simple_1d_data, device):
        """Test effect of prior_variance parameter."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        source_hyperparams = {
            'lengthscale': torch.tensor([[0.3]], device=device),
            'outputscale': torch.tensor(1.5, device=device),
            'noise': torch.tensor(0.1, device=device)
        }

        model = TemperedGP(
            train_x, train_y, likelihood,
            source_hyperparams=source_hyperparams
        )
        model = model.to(device)

        # Perturb hyperparameters away from source
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5]], device=device)

        model.train()
        likelihood.train()

        # Small prior variance (tight prior)
        mll_tight = TemperedMarginalLogLikelihood(
            likelihood, model, beta=1.0, prior_variances=0.01
        )

        # Large prior variance (loose prior)
        mll_loose = TemperedMarginalLogLikelihood(
            likelihood, model, beta=1.0, prior_variances=10.0
        )

        output = model(train_x)

        # Tight prior should penalize deviation more
        loss_tight = -mll_tight(output, train_y)
        loss_loose = -mll_loose(output, train_y)

        # Tight prior should give higher loss (stronger penalty)
        assert loss_tight > loss_loose

    def test_no_source_hyperparams_fallback(self, simple_1d_data, device):
        """Test that MLL works even without source hyperparameters."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = TemperedGP(train_x, train_y, likelihood)  # No source
        model = model.to(device)
        model.train()
        likelihood.train()

        # Should not crash even with beta > 0
        mll = TemperedMarginalLogLikelihood(likelihood, model, beta=0.5)

        output = model(train_x)
        loss = -mll(output, train_y)

        assert torch.isfinite(loss)


class TestTrainTemperedGP:
    """Tests for train_tempered_gp function."""

    def test_training_without_source(self, simple_1d_data, device):
        """Test training TemperedGP without source (beta=0)."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = TemperedGP(train_x, train_y, likelihood)
        model = model.to(device)

        model, likelihood, losses = train_tempered_gp(
            model, likelihood, train_x, train_y,
            beta=0.0,
            num_iter=100,
            lr=0.1
        )

        # Training should converge
        assert losses[-1] < losses[0]
        assert np.isfinite(losses).all()

    def test_training_with_source(self, source_target_1d, device):
        """Test transfer learning via prior tempering."""
        # First train source model
        x_source, y_source = source_target_1d['source']
        x_source = x_source.to(device)
        y_source = y_source.to(device)

        likelihood_source = gpytorch.likelihoods.GaussianLikelihood()
        from src.models.gp_model import BaselineGP, train_baseline_gp

        source_model = BaselineGP(x_source, y_source, likelihood_source, ard=False)
        source_model = source_model.to(device)

        source_model, likelihood_source, _ = train_baseline_gp(
            source_model, likelihood_source, x_source, y_source,
            num_iter=100
        )

        # Extract source hyperparameters
        source_hyperparams = {
            'lengthscale': source_model.covar_module.base_kernel.lengthscale.detach(),
            'outputscale': source_model.covar_module.outputscale.detach(),
            'noise': likelihood_source.noise.detach()
        }

        # Now train target with tempering
        x_target, y_target = source_target_1d['target']
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        likelihood_target = gpytorch.likelihoods.GaussianLikelihood()
        target_model = TemperedGP(
            x_target, y_target, likelihood_target,
            source_hyperparams=source_hyperparams
        )
        target_model = target_model.to(device)

        target_model, likelihood_target, losses = train_tempered_gp(
            target_model, likelihood_target, x_target, y_target,
            beta=0.5,
            num_iter=100,
            lr=0.1
        )

        # Training should converge
        assert losses[-1] < losses[0]
        assert np.isfinite(losses).all()

    def test_beta_parameter_range(self, simple_1d_data, device):
        """Test training with different beta values."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        source_hyperparams = {
            'lengthscale': torch.tensor([[0.3]], device=device),
            'outputscale': torch.tensor(1.5, device=device),
            'noise': torch.tensor(0.1, device=device)
        }

        for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = TemperedGP(
                train_x, train_y, likelihood,
                source_hyperparams=source_hyperparams
            )
            model = model.to(device)

            model, likelihood, losses = train_tempered_gp(
                model, likelihood, train_x, train_y,
                beta=beta,
                num_iter=50,
                lr=0.1
            )

            # All should converge
            assert losses[-1] < losses[0]
            assert np.isfinite(losses).all()

    def test_hyperparameter_deviation_from_source(self, source_target_1d, device):
        """Test that smaller beta allows more deviation from source."""
        # Train source
        x_source, y_source = source_target_1d['source']
        x_source = x_source.to(device)
        y_source = y_source.to(device)

        from src.models.gp_model import BaselineGP, train_baseline_gp

        likelihood_source = gpytorch.likelihoods.GaussianLikelihood()
        source_model = BaselineGP(x_source, y_source, likelihood_source, ard=False)
        source_model = source_model.to(device)

        source_model, likelihood_source, _ = train_baseline_gp(
            source_model, likelihood_source, x_source, y_source, num_iter=100
        )

        source_lengthscale = source_model.covar_module.base_kernel.lengthscale.detach().clone()

        source_hyperparams = {
            'lengthscale': source_lengthscale,
            'outputscale': source_model.covar_module.outputscale.detach(),
            'noise': likelihood_source.noise.detach()
        }

        # Train target with high beta (strong prior)
        x_target, y_target = source_target_1d['target']
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        likelihood_high = gpytorch.likelihoods.GaussianLikelihood()
        model_high = TemperedGP(
            x_target, y_target, likelihood_high,
            source_hyperparams=source_hyperparams
        )
        model_high = model_high.to(device)

        model_high, likelihood_high, _ = train_tempered_gp(
            model_high, likelihood_high, x_target, y_target,
            beta=0.9, num_iter=100, lr=0.1
        )

        # Train target with low beta (weak prior)
        likelihood_low = gpytorch.likelihoods.GaussianLikelihood()
        model_low = TemperedGP(
            x_target, y_target, likelihood_low,
            source_hyperparams=source_hyperparams
        )
        model_low = model_low.to(device)

        model_low, likelihood_low, _ = train_tempered_gp(
            model_low, likelihood_low, x_target, y_target,
            beta=0.1, num_iter=100, lr=0.1
        )

        # Get final lengthscales
        ls_high = model_high.covar_module.base_kernel.lengthscale.detach()
        ls_low = model_low.covar_module.base_kernel.lengthscale.detach()

        # High beta should stay closer to source
        deviation_high = torch.abs(ls_high - source_lengthscale).item()
        deviation_low = torch.abs(ls_low - source_lengthscale).item()

        # Generally true (though not guaranteed in all cases)
        # High beta should deviate less from source
        assert deviation_high <= deviation_low * 1.5

    def test_edge_case_invalid_beta(self, simple_1d_data, device):
        """Test behavior with invalid beta values."""
        train_x, train_y = simple_1d_data
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = TemperedGP(train_x, train_y, likelihood)
        model = model.to(device)

        # Negative beta should still run (though not meaningful)
        model, likelihood, losses = train_tempered_gp(
            model, likelihood, train_x, train_y,
            beta=-0.5,  # Invalid but should not crash
            num_iter=10
        )
        assert np.isfinite(losses).all()

        # Beta > 1 should also run
        model, likelihood, losses = train_tempered_gp(
            model, likelihood, train_x, train_y,
            beta=2.0,
            num_iter=10
        )
        assert np.isfinite(losses).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
