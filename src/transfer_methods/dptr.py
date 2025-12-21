"""
Deep Probabilistic Transfer Regression (DPTR)

Reference: Chai et al. (2021)
"Deep Probabilistic Transfer Learning for Missing Data Scenarios"

Uses VAE to align latent representations between source and target domains,
enabling transfer learning with missing features or sensor differences.
"""

import torch
import torch.nn as nn
import gpytorch
import numpy as np
from typing import Tuple, Optional


class FeatureEncoder(nn.Module):
    """Encode features to latent space."""

    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class FeatureDecoder(nn.Module):
    """Decode from latent space to feature space."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return self.fc2(h)


class DPTRVAE(nn.Module):
    """
    Variational Autoencoder for domain alignment.

    Learns shared latent representation between source and target features,
    handling missing data and feature mismatches.
    """

    def __init__(self,
                 source_dim: int,
                 target_dim: int,
                 latent_dim: int = 10,
                 hidden_dim: int = 32):
        super().__init__()

        self.source_encoder = FeatureEncoder(source_dim, latent_dim, hidden_dim)
        self.target_encoder = FeatureEncoder(target_dim, latent_dim, hidden_dim)
        self.source_decoder = FeatureDecoder(latent_dim, source_dim, hidden_dim)
        self.target_decoder = FeatureDecoder(latent_dim, target_dim, hidden_dim)

        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_source(self, x):
        """Encode source features to latent space."""
        mu, logvar = self.source_encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode_target(self, x):
        """Encode target features to latent space."""
        mu, logvar = self.target_encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_source(self, z):
        """Decode latent to source features."""
        return self.source_decoder(z)

    def decode_target(self, z):
        """Decode latent to target features."""
        return self.target_decoder(z)

    def forward(self, x_source=None, x_target=None):
        """
        Forward pass for both domains.

        Returns reconstructions and latent distributions.
        """
        results = {}

        if x_source is not None:
            z_s, mu_s, logvar_s = self.encode_source(x_source)
            x_s_recon = self.decode_source(z_s)
            results['source'] = {
                'z': z_s, 'mu': mu_s, 'logvar': logvar_s, 'recon': x_s_recon
            }

        if x_target is not None:
            z_t, mu_t, logvar_t = self.encode_target(x_target)
            x_t_recon = self.decode_target(z_t)
            results['target'] = {
                'z': z_t, 'mu': mu_t, 'logvar': logvar_t, 'recon': x_t_recon
            }

        return results


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss: reconstruction + KL divergence.

    Args:
        recon_x: Reconstructed features
        x: Original features
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL term (β-VAE)
    """
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


def train_vae(vae: DPTRVAE,
              X_source: torch.Tensor,
              X_target: torch.Tensor,
              n_epochs: int = 100,
              lr: float = 1e-3,
              beta: float = 0.5,
              verbose: bool = False):
    """
    Train VAE for domain alignment.

    Args:
        vae: DPTRVAE model
        X_source: Source features [N_s, D_s]
        X_target: Target features [N_t, D_t]
        n_epochs: Training epochs
        lr: Learning rate
        beta: β-VAE parameter
        verbose: Print progress
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = vae(X_source, X_target)

        # Compute losses
        loss_s = vae_loss(
            outputs['source']['recon'], X_source,
            outputs['source']['mu'], outputs['source']['logvar'],
            beta=beta
        )
        loss_t = vae_loss(
            outputs['target']['recon'], X_target,
            outputs['target']['mu'], outputs['target']['logvar'],
            beta=beta
        )

        # Total loss
        loss = loss_s + loss_t

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    return vae


class DPTRGaussianProcess(gpytorch.models.ExactGP):
    """
    GP that operates in VAE-aligned latent space.

    Uses learned latent representations for predictions.
    """

    def __init__(self,
                 train_z: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood):
        super().__init__(train_z, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_z.shape[-1])
        )

    def forward(self, z):
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_z, covar_z)


def train_dptr_gp(source_x: torch.Tensor,
                  source_y: torch.Tensor,
                  target_x: torch.Tensor,
                  target_y: torch.Tensor,
                  latent_dim: int = 10,
                  hidden_dim: int = 32,
                  vae_epochs: int = 100,
                  gp_epochs: int = 50,
                  beta: float = 0.5,
                  verbose: bool = False) -> Tuple:
    """
    Train DPTR: VAE + GP in latent space.

    Args:
        source_x: Source features [N_s, D_s]
        source_y: Source targets [N_s]
        target_x: Target features [N_t, D_t]
        target_y: Target targets [N_t]
        latent_dim: Latent space dimension
        hidden_dim: VAE hidden layer size
        vae_epochs: VAE training epochs
        gp_epochs: GP training epochs
        beta: β-VAE parameter
        verbose: Print progress

    Returns:
        vae: Trained VAE
        gp_model: GP in latent space
        likelihood: GP likelihood
        dptr_info: Training information
    """
    # Initialize and train VAE
    vae = DPTRVAE(
        source_dim=source_x.shape[-1],
        target_dim=target_x.shape[-1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    )

    if verbose:
        print("Training VAE for domain alignment...")

    vae = train_vae(
        vae, source_x, target_x,
        n_epochs=vae_epochs,
        beta=beta,
        verbose=verbose
    )

    # Encode to latent space
    vae.eval()
    with torch.no_grad():
        z_source, _, _ = vae.encode_source(source_x)
        z_target, _, _ = vae.encode_target(target_x)

    # Train GP in latent space on source data
    if verbose:
        print("Training GP in latent space...")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = DPTRGaussianProcess(z_source, source_y, likelihood)

    from src.models.gp_model import train_baseline_gp
    gp_model, likelihood = train_baseline_gp(
        gp_model, likelihood, z_source, source_y,
        num_iter=gp_epochs,
        verbose=verbose
    )

    dptr_info = {
        'latent_dim': latent_dim,
        'z_source': z_source.detach().cpu().numpy(),
        'z_target': z_target.detach().cpu().numpy(),
        'vae_beta': beta,
    }

    return vae, gp_model, likelihood, dptr_info


def predict_dptr(vae: DPTRVAE,
                gp_model: DPTRGaussianProcess,
                likelihood: gpytorch.likelihoods.Likelihood,
                target_x: torch.Tensor,
                source_domain: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using DPTR.

    Args:
        vae: Trained VAE
        gp_model: Trained GP
        likelihood: GP likelihood
        target_x: Features to predict [N, D]
        source_domain: If True, use source encoder

    Returns:
        predictions: Mean predictions
        uncertainties: Prediction std
    """
    vae.eval()
    gp_model.eval()

    with torch.no_grad():
        # Encode to latent space
        if source_domain:
            z, _, _ = vae.encode_source(target_x)
        else:
            z, _, _ = vae.encode_target(target_x)

        # GP prediction in latent space
        f_dist = gp_model(z)
        pred_dist = likelihood(f_dist)

        predictions = pred_dist.mean.cpu().numpy()
        uncertainties = pred_dist.stddev.cpu().numpy()

    return predictions, uncertainties
