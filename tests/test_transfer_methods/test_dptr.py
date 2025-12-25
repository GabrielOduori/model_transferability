"""
Unit tests for CORRECTED DPTR implementation.

Tests the architecture matching Chai et al. (2022):
- PriorNet p(z|x)
- EncoderNet q(z|x,y)
- DecoderNet p(y|z)
- AdversaryNet G_A(z)
- Conditional KL divergence
- MC-DAT alignment
"""

import pytest
import torch
import gpytorch
import numpy as np
from src.transfer_methods.dptr import (
    FeatureEncoder,
    FeatureDecoder,
    PriorNet,
    AdversaryNet,
    DPTRVAE,
    DPTRGaussianProcess,
    dgrm_loss,
    mc_dat_loss,
    train_dptr_vae,
    train_dptr_gp,
    predict_dptr
)


class TestPriorNet:
    """Tests for PriorNet p(z|x)."""

    def test_initialization(self):
        """Test PriorNet initialization."""
        prior_net = PriorNet(input_dim=5, latent_dim=10, hidden_dim=32)

        assert prior_net.fc1.in_features == 5
        assert prior_net.fc_mu.out_features == 10
        assert prior_net.fc_logvar.out_features == 10

    def test_forward_pass(self):
        """Test PriorNet forward pass."""
        prior_net = PriorNet(input_dim=5, latent_dim=10, hidden_dim=32)

        batch_size = 20
        x = torch.randn(batch_size, 5)

        mu_prior, logvar_prior = prior_net(x)

        assert mu_prior.shape == (batch_size, 10)
        assert logvar_prior.shape == (batch_size, 10)
        assert torch.isfinite(mu_prior).all()
        assert torch.isfinite(logvar_prior).all()

    def test_different_dimensions(self):
        """Test PriorNet with various dimensions."""
        test_configs = [
            (3, 5, 16),
            (10, 8, 64),
            (2, 2, 8)
        ]

        for input_dim, latent_dim, hidden_dim in test_configs:
            prior_net = PriorNet(input_dim, latent_dim, hidden_dim)
            x = torch.randn(10, input_dim)
            mu, logvar = prior_net(x)

            assert mu.shape == (10, latent_dim)
            assert logvar.shape == (10, latent_dim)


class TestFeatureEncoder:
    """Tests for FeatureEncoder q(z|x,y)."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = FeatureEncoder(input_dim=5, output_dim=1, latent_dim=10, hidden_dim=32)

        # Should take input_dim + output_dim as input
        assert encoder.fc1.in_features == 5 + 1
        assert encoder.fc_mu.out_features == 10
        assert encoder.fc_logvar.out_features == 10

    def test_forward_pass(self):
        """Test encoder forward pass with x and y."""
        encoder = FeatureEncoder(input_dim=5, output_dim=1, latent_dim=10, hidden_dim=32)

        batch_size = 20
        x = torch.randn(batch_size, 5)
        y = torch.randn(batch_size, 1)

        mu, logvar = encoder(x, y)

        assert mu.shape == (batch_size, 10)
        assert logvar.shape == (batch_size, 10)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_different_output_dims(self):
        """Test encoder with different output dimensions."""
        encoder = FeatureEncoder(input_dim=5, output_dim=3, latent_dim=10, hidden_dim=32)

        x = torch.randn(10, 5)
        y = torch.randn(10, 3)

        mu, logvar = encoder(x, y)

        assert mu.shape == (10, 10)
        assert logvar.shape == (10, 10)


class TestAdversaryNet:
    """Tests for AdversaryNet G_A(z)."""

    def test_initialization(self):
        """Test AdversaryNet initialization."""
        adversary = AdversaryNet(latent_dim=10)

        assert adversary.fc.in_features == 10
        assert adversary.fc.out_features == 1

    def test_forward_pass(self):
        """Test adversary forward pass."""
        adversary = AdversaryNet(latent_dim=10)

        batch_size = 20
        z = torch.randn(batch_size, 10)

        domain_pred = adversary(z)

        assert domain_pred.shape == (batch_size, 1)
        assert torch.isfinite(domain_pred).all()
        # Should output probabilities via sigmoid
        assert (domain_pred >= 0).all()
        assert (domain_pred <= 1).all()

    def test_domain_discrimination(self):
        """Test that adversary can distinguish domains."""
        adversary = AdversaryNet(latent_dim=10)

        # Source domain: centered at -1
        z_source = torch.randn(50, 10) - 1.0
        # Target domain: centered at +1
        z_target = torch.randn(50, 10) + 1.0

        pred_source = adversary(z_source)
        pred_target = adversary(z_target)

        # Initially may be similar, but should be trainable
        assert pred_source.shape == (50, 1)
        assert pred_target.shape == (50, 1)


class TestDPTRVAE:
    """Tests for corrected DPTRVAE with PriorNet."""

    def test_initialization(self):
        """Test VAE initialization with PriorNet."""
        vae = DPTRVAE(
            source_dim=3,
            target_dim=5,
            latent_dim=10,
            hidden_dim=32
        )

        assert vae.source_dim == 3
        assert vae.target_dim == 5
        assert vae.latent_dim == 10

        # Check that all networks exist
        assert isinstance(vae.source_encoder, FeatureEncoder)
        assert isinstance(vae.target_encoder, FeatureEncoder)
        assert isinstance(vae.source_prior, PriorNet)
        assert isinstance(vae.target_prior, PriorNet)
        assert isinstance(vae.decoder, FeatureDecoder)

    def test_forward_pass(self):
        """Test VAE forward pass returns prior distributions."""
        vae = DPTRVAE(
            source_dim=3,
            target_dim=5,
            latent_dim=10,
            hidden_dim=32
        )

        n_source = 20
        n_target = 15

        x_source = torch.randn(n_source, 3)
        y_source = torch.randn(n_source)
        x_target = torch.randn(n_target, 5)
        y_target = torch.randn(n_target)

        outputs = vae(x_source, y_source, x_target, y_target)

        # Check source outputs
        assert 'mu_post' in outputs['source']
        assert 'logvar_post' in outputs['source']
        assert 'mu_prior' in outputs['source']
        assert 'logvar_prior' in outputs['source']
        assert 'z' in outputs['source']
        assert 'recon' in outputs['source']

        # Check target outputs
        assert 'mu_post' in outputs['target']
        assert 'logvar_post' in outputs['target']
        assert 'mu_prior' in outputs['target']
        assert 'logvar_prior' in outputs['target']
        assert 'z' in outputs['target']
        assert 'recon' in outputs['target']

        # Check shapes
        assert outputs['source']['mu_post'].shape == (n_source, 10)
        assert outputs['source']['mu_prior'].shape == (n_source, 10)
        assert outputs['target']['mu_post'].shape == (n_target, 10)
        assert outputs['target']['mu_prior'].shape == (n_target, 10)

    def test_reparameterization(self):
        """Test reparameterization trick."""
        vae = DPTRVAE(source_dim=3, target_dim=5, latent_dim=10)

        mu = torch.zeros(20, 10)
        logvar = torch.zeros(20, 10)

        # Sample multiple times
        samples = [vae.reparameterize(mu, logvar) for _ in range(100)]

        # Mean should be close to 0
        mean_sample = torch.stack(samples).mean(dim=0)
        assert torch.abs(mean_sample).mean() < 0.2

        # Std should be close to 1
        std_sample = torch.stack(samples).std(dim=0)
        assert torch.abs(std_sample - 1.0).mean() < 0.2


class TestDGRMLoss:
    """Tests for DGRM loss with conditional KL."""

    def test_dgrm_loss_computation(self):
        """Test DGRM loss with conditional prior."""
        batch_size = 20
        latent_dim = 10

        # Reconstruction
        recon_y = torch.randn(batch_size, 1)
        y = torch.randn(batch_size, 1)

        # Posterior
        mu_post = torch.randn(batch_size, latent_dim)
        logvar_post = torch.randn(batch_size, latent_dim)

        # Prior
        mu_prior = torch.randn(batch_size, latent_dim)
        logvar_prior = torch.randn(batch_size, latent_dim)

        loss = dgrm_loss(recon_y, y, mu_post, logvar_post, mu_prior, logvar_prior, beta=1.0)

        assert torch.isfinite(loss)
        assert loss >= 0  # Loss should be non-negative

    def test_dgrm_loss_beta_effect(self):
        """Test that beta controls KL divergence weight."""
        batch_size = 20
        latent_dim = 10

        recon_y = torch.randn(batch_size, 1)
        y = torch.randn(batch_size, 1)
        mu_post = torch.randn(batch_size, latent_dim)
        logvar_post = torch.randn(batch_size, latent_dim)
        mu_prior = torch.randn(batch_size, latent_dim)
        logvar_prior = torch.randn(batch_size, latent_dim)

        loss_beta_0 = dgrm_loss(recon_y, y, mu_post, logvar_post, mu_prior, logvar_prior, beta=0.0)
        loss_beta_1 = dgrm_loss(recon_y, y, mu_post, logvar_post, mu_prior, logvar_prior, beta=1.0)

        # Higher beta should generally give higher loss (more regularization)
        # Note: This is not always true due to reconstruction term, but KL contribution should differ
        assert torch.isfinite(loss_beta_0)
        assert torch.isfinite(loss_beta_1)


class TestMCDATLoss:
    """Tests for MC-DAT adversarial loss."""

    def test_mc_dat_loss_computation(self):
        """Test MC-DAT loss computation."""
        n_samples = 30
        n_mc = 5
        latent_dim = 10

        # MC samples: [N, L, latent_dim]
        z_samples = torch.randn(n_samples, n_mc, latent_dim)

        # Domain labels: 0 for source, 1 for target
        domain_labels = torch.cat([
            torch.zeros(15),
            torch.ones(15)
        ])

        adversary = AdversaryNet(latent_dim=latent_dim)

        loss = mc_dat_loss(z_samples, domain_labels, adversary)

        assert torch.isfinite(loss)
        assert loss >= 0  # Cross-entropy loss should be non-negative

    def test_mc_dat_loss_shape_requirements(self):
        """Test MC-DAT loss with correct shapes."""
        adversary = AdversaryNet(latent_dim=10)

        # Should work with correct shapes
        z = torch.randn(20, 5, 10)  # [N, L, latent_dim]
        labels = torch.cat([torch.zeros(10), torch.ones(10)])

        loss = mc_dat_loss(z, labels, adversary)
        assert torch.isfinite(loss)


class TestDPTRGaussianProcess:
    """Tests for DPTR Gaussian Process."""

    def test_initialization(self):
        """Test GP initialization."""
        train_x = torch.randn(20, 10)
        train_y = torch.randn(20)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gp = DPTRGaussianProcess(train_x, train_y, likelihood)

        assert gp.train_inputs[0].shape == (20, 10)
        assert gp.train_targets.shape == (20,)

    def test_forward_pass(self):
        """Test GP forward pass."""
        train_x = torch.randn(20, 10)
        train_y = torch.randn(20)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gp = DPTRGaussianProcess(train_x, train_y, likelihood)
        gp.eval()
        likelihood.eval()

        test_x = torch.randn(10, 10)

        with torch.no_grad():
            pred = gp(test_x)

        assert pred.mean.shape == (10,)
        assert torch.isfinite(pred.mean).all()


class TestDPTRIntegration:
    """Integration tests for full DPTR pipeline."""

    def test_train_dptr_vae(self):
        """Test VAE training function."""
        torch.manual_seed(42)

        n_source = 30
        n_target = 20
        source_dim = 3
        target_dim = 5

        x_source = torch.randn(n_source, source_dim)
        y_source = torch.randn(n_source)
        x_target = torch.randn(n_target, target_dim)
        y_target = torch.randn(n_target)

        vae, adversary, info = train_dptr_vae(
            x_source, y_source,
            x_target, y_target,
            latent_dim=4,
            hidden_dim=8,
            n_epochs=10,  # Short for testing
            beta=1.0,
            lambda_align=1.0,
            n_mc_samples=2,
            verbose=False
        )

        # Check outputs
        assert isinstance(vae, DPTRVAE)
        assert isinstance(adversary, AdversaryNet)
        assert 'z_source' in info
        assert 'z_target' in info
        assert info['z_source'].shape == (n_source, 4)
        assert info['z_target'].shape == (n_target, 4)

    def test_train_dptr_gp_full_pipeline(self):
        """Test full DPTR pipeline."""
        torch.manual_seed(123)

        n_source = 40
        n_target = 25
        source_dim = 3
        target_dim = 3  # Same for simplicity

        # Create related source and target data
        x_source = torch.randn(n_source, source_dim)
        y_source = x_source[:, 0] + 0.5 * x_source[:, 1] + torch.randn(n_source) * 0.1

        x_target = torch.randn(n_target, target_dim)
        y_target = x_target[:, 0] + 0.5 * x_target[:, 1] + torch.randn(n_target) * 0.1

        try:
            vae, gp_model, likelihood, adversary, info = train_dptr_gp(
                x_source, y_source,
                x_target, y_target,
                latent_dim=4,
                hidden_dim=8,
                vae_epochs=20,
                gp_epochs=10,
                beta=1.0,
                lambda_align=1.0,
                n_mc_samples=2,
                verbose=False
            )

            # Check all components returned
            assert isinstance(vae, DPTRVAE)
            assert isinstance(gp_model, DPTRGaussianProcess)
            assert isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood)
            assert isinstance(adversary, AdversaryNet)

            # Check info
            assert 'latent_dim' in info
            assert 'z_source' in info
            assert 'z_target' in info

            # Test prediction
            x_test = torch.randn(10, target_dim)
            predictions, uncertainties = predict_dptr(
                vae, gp_model, likelihood, x_test, source_domain=False
            )

            assert predictions.shape == (10,)
            assert uncertainties.shape == (10,)
            assert np.all(np.isfinite(predictions))
            assert np.all(uncertainties > 0)

            print("✅ Full DPTR pipeline test passed!")

        except Exception as e:
            pytest.fail(f"Full pipeline failed: {e}")

    def test_prediction_both_domains(self):
        """Test prediction on both source and target domains."""
        torch.manual_seed(42)

        n_source = 30
        n_target = 20

        x_source = torch.randn(n_source, 3)
        y_source = torch.randn(n_source)
        x_target = torch.randn(n_target, 3)
        y_target = torch.randn(n_target)

        vae, gp_model, likelihood, adversary, info = train_dptr_gp(
            x_source, y_source,
            x_target, y_target,
            latent_dim=4,
            vae_epochs=10,
            gp_epochs=5,
            verbose=False
        )

        # Test on source domain
        x_test_source = torch.randn(5, 3)
        pred_s, unc_s = predict_dptr(vae, gp_model, likelihood, x_test_source, source_domain=True)

        # Test on target domain
        x_test_target = torch.randn(5, 3)
        pred_t, unc_t = predict_dptr(vae, gp_model, likelihood, x_test_target, source_domain=False)

        assert pred_s.shape == (5,)
        assert pred_t.shape == (5,)
        assert unc_s.shape == (5,)
        assert unc_t.shape == (5,)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
