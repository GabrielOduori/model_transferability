"""
Deep Probabilistic Transfer Regression (DPTR)

Reference: Chai et al. (2022)
"A Deep Probabilistic Transfer Learning Framework for Soft Sensor Modeling With Missing Data"
IEEE Transactions on Neural Networks and Learning Systems

Uses VAE with conditional prior p(z|x) to align latent representations between
source and target domains, enabling transfer learning with missing features.

CORRECTED IMPLEMENTATION (2025-12-24):
1. Added PriorNet to learn conditional prior p(z|x) per equation (9)
2. Fixed KL divergence to use conditional prior formula
3. Replaced MMD with Adversarial Discriminator (MC-DAT) per equation (13)
"""

import torch
import torch.nn as nn
import gpytorch
import numpy as np
from typing import Tuple, Optional


class FeatureEncoder(nn.Module):
    """
    Encoder network q_φ(z|x,y) - EncoderNet in the paper.

    Encodes features x and output y to posterior distribution over latent z.
    """

    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        # Concatenate input features x and output y
        self.fc1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        """
        Args:
            x: Input features [N, input_dim]
            y: Output values [N, output_dim]

        Returns:
            mu: Posterior mean [N, latent_dim]
            logvar: Posterior log variance [N, latent_dim]
        """
        xy = torch.cat([x, y.unsqueeze(-1) if y.dim() == 1 else y], dim=-1)
        h = torch.relu(self.fc1(xy))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class PriorNet(nn.Module):
    """
    Prior network p_θ(z|x) - PriorNet in the paper.

    Learns conditional prior distribution over z given only input x.
    This is CRITICAL for DPTR and was missing in the original implementation!
    """

    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input features [N, input_dim]

        Returns:
            mu: Prior mean μ_x [N, latent_dim]
            logvar: Prior log variance log(σ²_x) [N, latent_dim]
        """
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class FeatureDecoder(nn.Module):
    """
    Decoder network p_θ(y|z) - DecoderNet in the paper.

    Decodes from latent space z to output prediction y.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: Latent codes [N, latent_dim]

        Returns:
            y: Output predictions [N, output_dim]
        """
        h = torch.relu(self.fc1(z))
        return self.fc2(h)


class AdversaryNet(nn.Module):
    """
    Domain discriminator G_A(W_A; z) for adversarial training.

    Predicts whether latent code z comes from source (0) or target (1) domain.
    Used in Monte Carlo Domain Adversarial Training (MC-DAT).
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, z):
        """
        Args:
            z: Latent codes [N, latent_dim]

        Returns:
            prob: Probability of being target domain [N, 1]
        """
        return torch.sigmoid(self.fc(z))


class DPTRVAE(nn.Module):
    """
    Deep Generative Regression Model (DGRM) for DPTR.

    Architecture (from paper Section III.A):
    - EncoderNet: q_φ(z|x,y) - posterior over latent z
    - PriorNet: p_θ(z|x) - conditional prior (CRITICAL!)
    - DecoderNet: p_θ(y|z) - output prediction

    Handles different source and target feature dimensions.
    """

    def __init__(self,
                 source_dim: int,
                 target_dim: int,
                 latent_dim: int = 10,
                 hidden_dim: int = 32):
        super().__init__()

        self.source_dim = source_dim
        self.target_dim = target_dim
        # Posterior: q_φ(z|x,y)
        self.source_encoder = FeatureEncoder(source_dim, 1, latent_dim, hidden_dim)
        self.target_encoder = FeatureEncoder(target_dim, 1, latent_dim, hidden_dim)

        # Conditional Prior: p_θ(z|x) - THIS WAS MISSING!
        self.source_prior = PriorNet(source_dim, latent_dim, hidden_dim)
        self.target_prior = PriorNet(target_dim, latent_dim, hidden_dim)

        # Decoder: p_θ(y|z)
        self.decoder = FeatureDecoder(latent_dim, 1, hidden_dim)

        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_source(self, x, y=None):
        """
        Encode source (x,y) to latent posterior.

        Returns:
            z: Sampled latent code
            mu_post: Posterior mean μ_xy
            logvar_post: Posterior log variance log(σ²_xy)
        """
        provided = y is not None
        if not provided:
            y = torch.zeros(x.shape[0], 1, device=x.device)
        mu_post, logvar_post = self.source_encoder(x, y)
        z = self.reparameterize(mu_post, logvar_post)
        return (z, mu_post, logvar_post) if provided else z

    def encode_target(self, x, y=None):
        """Encode target (x,y) to latent posterior."""
        if y is None:
            y = torch.zeros(x.shape[0], 1, device=x.device)
            mu_post, logvar_post = self.target_prior(x)
            z = self.reparameterize(mu_post, logvar_post)
            return z

        mu_post, logvar_post = self.target_encoder(x, y)
        z = self.reparameterize(mu_post, logvar_post)
        return z, mu_post, logvar_post

    def prior_source(self, x):
        """
        Get conditional prior for source: p_θ(z|x).

        Returns:
            mu_prior: Prior mean μ_x
            logvar_prior: Prior log variance log(σ²_x)
        """
        return self.source_prior(x)

    def prior_target(self, x):
        """Get conditional prior for target: p_θ(z|x)."""
        return self.target_prior(x)

    def decode(self, z):
        """Decode latent to output: p_θ(y|z)."""
        return self.decoder(z)

    def forward(self, x_source=None, y_source=None, x_target=None, y_target=None):
        """
        Forward pass for both domains.

        Returns dict with posterior, prior, reconstruction for each domain.
        """
        results = {}

        if x_source is not None and y_source is not None:
            z_s, mu_post_s, logvar_post_s = self.encode_source(x_source, y_source)
            mu_prior_s, logvar_prior_s = self.prior_source(x_source)
            y_s_recon = self.decode(z_s)

            results['source'] = {
                'z': z_s,
                'mu_post': mu_post_s,
                'logvar_post': logvar_post_s,
                'mu_prior': mu_prior_s,
                'logvar_prior': logvar_prior_s,
                'recon': y_s_recon
            }

        if x_target is not None and y_target is not None:
            z_t, mu_post_t, logvar_post_t = self.encode_target(x_target, y_target)
            mu_prior_t, logvar_prior_t = self.prior_target(x_target)
            y_t_recon = self.decode(z_t)

            results['target'] = {
                'z': z_t,
                'mu_post': mu_post_t,
                'logvar_post': logvar_post_t,
                'mu_prior': mu_prior_t,
                'logvar_prior': logvar_prior_t,
                'recon': y_t_recon
            }

        return results


def dgrm_loss(recon_y, y, mu_post, logvar_post, mu_prior, logvar_prior, beta=1.0):
    """
    Deep Generative Regression Model (DGRM) loss per equation (10).

    L_DGRM = KL[q_φ(z|x,y) || p_θ(z|x)] + E[log p_θ(y|z)]

    Uses CONDITIONAL prior p_θ(z|x), not standard N(0,I)!

    Args:
        recon_y: Reconstructed output
        y: Original output
        mu_post: Posterior mean μ_xy from encoder
        logvar_post: Posterior log variance from encoder
        mu_prior: Prior mean μ_x from PriorNet
        logvar_prior: Prior log variance from PriorNet
        beta: Weight for KL term (β-VAE)

    Returns:
        loss: DGRM loss value
    """
    # Reconstruction loss: -E[log p_θ(y|z)]
    recon_loss = nn.functional.mse_loss(recon_y, y, reduction='sum')

    # KL divergence: KL[q_φ(z|x,y) || p_θ(z|x)]
    # Formula from equation (9):
    # KL = 0.5 * Σ[log(σ²_prior/σ²_post) - σ²_post/σ²_prior - (μ_post-μ_prior)²/σ²_prior + 1]

    var_post = logvar_post.exp()
    var_prior = logvar_prior.exp()

    kl_loss = 0.5 * torch.sum(
        logvar_prior - logvar_post           # log(σ²_prior/σ²_post)
        - var_post / (var_prior + 1e-8)      # -σ²_post/σ²_prior
        - (mu_post - mu_prior).pow(2) / (var_prior + 1e-8)  # -(μ_post-μ_prior)²/σ²_prior
        + 1
    )

    loss = recon_loss + beta * kl_loss
    return torch.clamp(loss, min=0.0)


def mc_dat_loss(z_samples, domain_labels, adversary_net):
    """
    Monte Carlo Domain Adversarial Training (MC-DAT) loss per equation (13).

    L_MC-DAT = -1/L Σ[a^i log G_A(z^i_l) + (1-a^i)log(1-G_A(z^i_l))]

    This is the CORRECT alignment loss from the paper (not MMD!).

    Args:
        z_samples: Latent samples [N, L, latent_dim] where L is num MC samples
        domain_labels: Domain labels [N] where 0=source, 1=target
        adversary_net: Domain discriminator network

    Returns:
        loss: Adversarial alignment loss
    """
    N, L, latent_dim = z_samples.shape

    # Average over Monte Carlo samples
    loss = 0.0
    for l in range(L):
        z_l = z_samples[:, l, :]  # [N, latent_dim]
        pred = adversary_net(z_l).squeeze()  # [N]

        # Binary cross-entropy: -[a*log(pred) + (1-a)*log(1-pred)]
        loss += -torch.mean(
            domain_labels * torch.log(pred + 1e-8) +
            (1 - domain_labels) * torch.log(1 - pred + 1e-8)
        )

    return loss / L


def train_dptr_vae(vae_or_x_source,
                   adversary_net: Optional[AdversaryNet] = None,
                   X_source: Optional[torch.Tensor] = None,
                   y_source: Optional[torch.Tensor] = None,
                   X_target: Optional[torch.Tensor] = None,
                   y_target: Optional[torch.Tensor] = None,
                   n_epochs: int = 100,
                   lr: float = 1e-3,
                   beta: float = 0.5,
                   lambda_align: float = 1.0,
                   n_mc_samples: int = 1,
                   verbose: bool = False,
                   latent_dim: Optional[int] = None,
                   hidden_dim: int = 32,
                   **kwargs):
    """
    Train DPTR VAE with adversarial domain adaptation per equation (16).

    CORRECTED to match paper's architecture:
    1. Uses conditional prior p_θ(z|x) via PriorNet
    2. Uses correct KL divergence formula (equation 9)
    3. Uses adversarial discriminator (MC-DAT), not MMD
    4. Min-max optimization: maximize for VAE, minimize for adversary

    Args:
        vae: DPTRVAE model
        adversary_net: Domain discriminator
        X_source: Source features [N_s, D_s]
        y_source: Source outputs [N_s]
        X_target: Target features [N_t, D_t]
        y_target: Target outputs [N_t]
        n_epochs: Training epochs
        lr: Learning rate
        beta: β-VAE parameter (weight for KL divergence)
        lambda_align: Weight for alignment loss
        n_mc_samples: Number of Monte Carlo samples for MC-DAT
        verbose: Print progress

    Returns:
        vae: Trained VAE
        adversary_net: Trained adversary
    """
    # Support both (vae, adversary, data...) and (X_source, y_source, X_target, y_target, ...)
    if isinstance(vae_or_x_source, torch.Tensor):
        X_source, y_source, X_target, y_target = vae_or_x_source, adversary_net, X_source, y_source
        if latent_dim is None:
            latent_dim = max(X_source.shape[-1], X_target.shape[-1])
        vae = DPTRVAE(
            source_dim=X_source.shape[-1],
            target_dim=X_target.shape[-1],
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        adversary_net = AdversaryNet(latent_dim=latent_dim)
    else:
        vae = vae_or_x_source
        if latent_dim is None:
            latent_dim = vae.latent_dim
        if adversary_net is None:
            adversary_net = AdversaryNet(latent_dim=latent_dim)

    # Two optimizers: one for VAE (W_P, W_E, W_D), one for adversary (W_A)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)
    optimizer_adv = torch.optim.Adam(adversary_net.parameters(), lr=lr)

    # Domain labels: 0=source, 1=target
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    domain_labels = torch.cat([
        torch.zeros(n_source),
        torch.ones(n_target)
    ]).to(X_source.device)

    loss_history = []

    for epoch in range(n_epochs):
        # ===== Forward pass through DGRM =====
        outputs = vae(X_source, y_source, X_target, y_target)

        # ===== DGRM losses with conditional prior =====
        loss_s = dgrm_loss(
            outputs['source']['recon'],
            y_source.unsqueeze(-1) if y_source.dim() == 1 else y_source,
            outputs['source']['mu_post'],
            outputs['source']['logvar_post'],
            outputs['source']['mu_prior'],
            outputs['source']['logvar_prior'],
            beta=beta
        )

        loss_t = dgrm_loss(
            outputs['target']['recon'],
            y_target.unsqueeze(-1) if y_target.dim() == 1 else y_target,
            outputs['target']['mu_post'],
            outputs['target']['logvar_post'],
            outputs['target']['mu_prior'],
            outputs['target']['logvar_prior'],
            beta=beta
        )

        # ===== Sample latent codes for MC-DAT =====
        # Multiple Monte Carlo samples for stability
        z_samples = []
        for _ in range(n_mc_samples):
            # Sample from posterior using reparameterization
            z_s = vae.reparameterize(
                outputs['source']['mu_post'],
                outputs['source']['logvar_post']
            )
            z_t = vae.reparameterize(
                outputs['target']['mu_post'],
                outputs['target']['logvar_post']
            )
            z_samples.append(torch.cat([z_s, z_t], dim=0))

        z_samples = torch.stack(z_samples, dim=1)  # [N, L, latent_dim]

        # ===== Adversarial alignment loss (MC-DAT) =====
        loss_align = mc_dat_loss(z_samples, domain_labels, adversary_net)

        # ===== Total loss (equation 15) =====
        loss_dptr = loss_s + loss_t + lambda_align * loss_align
        loss_history.append(float(loss_dptr.detach()))

        # ===== Update VAE (MAXIMIZE L_DPTR) =====
        # Equation (16): argmax_{W_P,W_E,W_D} L_DPTR
        # Maximize = minimize negative
        optimizer_vae.zero_grad()
        optimizer_adv.zero_grad()  # Zero both before backward
        loss_vae = -loss_dptr  # Note the negative for maximization!
        loss_vae.backward(retain_graph=False)  # Don't retain graph
        optimizer_vae.step()

        # ===== Update Adversary (MINIMIZE L_MC-DAT) =====
        # Equation (16): argmin_{W_A} L_DPTR
        # Need to recompute forward pass for adversary update
        optimizer_vae.zero_grad()
        optimizer_adv.zero_grad()

        # Recompute forward and alignment loss
        outputs_adv = vae(X_source, y_source, X_target, y_target)
        z_samples_adv = []
        for _ in range(n_mc_samples):
            z_s = vae.reparameterize(
                outputs_adv['source']['mu_post'],
                outputs_adv['source']['logvar_post']
            )
            z_t = vae.reparameterize(
                outputs_adv['target']['mu_post'],
                outputs_adv['target']['logvar_post']
            )
            z_samples_adv.append(torch.cat([z_s, z_t], dim=0))
        z_samples_adv = torch.stack(z_samples_adv, dim=1)
        loss_align_adv = mc_dat_loss(z_samples_adv, domain_labels, adversary_net)

        loss_align_adv.backward()
        optimizer_adv.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  DGRM: source={loss_s.item():.4f}, target={loss_t.item():.4f}")
            print(f"  MC-DAT: {loss_align.item():.4f}")
            print(f"  Total: {loss_dptr.item():.4f}")

    vae.eval()
    with torch.no_grad():
        encoded_source = vae.encode_source(X_source, y_source)
        encoded_target = vae.encode_target(X_target, y_target)

        # encode_source returns tuple when y provided
        z_s = encoded_source[0] if isinstance(encoded_source, tuple) else encoded_source
        z_t = encoded_target[0] if isinstance(encoded_target, tuple) else encoded_target

    info = {
        'loss_history': loss_history,
        'z_source': z_s.detach().cpu().numpy(),
        'z_target': z_t.detach().cpu().numpy()
    }
    return vae, adversary_net, info


class DPTRGaussianProcess(gpytorch.models.ExactGP):
    """
    Wrapper that trains the DPTR VAE + GP and exposes a GP for predictions.
    """

    def __init__(
        self,
        train_z: Optional[torch.Tensor] = None,
        train_y: Optional[torch.Tensor] = None,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        source_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        latent_dim: int = 10,
        hidden_dim: int = 32,
        beta: float = 0.5,
        lambda_align: float = 1.0,
        n_mc_samples: int = 1
    ):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if train_z is None or train_y is None:
            dummy_z = torch.zeros(1, latent_dim)
            dummy_y = torch.zeros(1)
            super().__init__(dummy_z, dummy_y, likelihood)
        else:
            super().__init__(train_z, train_y, likelihood)

        feature_dim = train_z.shape[-1] if train_z is not None else latent_dim

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.lambda_align = lambda_align
        self.n_mc_samples = n_mc_samples

        # Will be set during training
        self.vae: Optional[DPTRVAE] = None
        self.adversary: Optional[AdversaryNet] = None
        self.gp_model: Optional[gpytorch.models.ExactGP] = None
        self.gp_likelihood: Optional[gpytorch.likelihoods.Likelihood] = None

    def forward(self, z):
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_z, covar_z)

    def fit_transfer(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
        y_target: torch.Tensor,
        vae_epochs: int = 50,
        gp_iterations: int = 50,
        device: Optional[torch.device] = None
    ):
        if device is not None:
            x_source = x_source.to(device)
            y_source = y_source.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

        vae, gp_model, gp_likelihood, adversary, _ = train_dptr_gp(
            x_source, y_source,
            x_target, y_target,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            vae_epochs=vae_epochs,
            gp_epochs=gp_iterations,
            beta=self.beta,
            lambda_align=self.lambda_align,
            n_mc_samples=self.n_mc_samples,
            verbose=False
        )

        self.vae = vae
        self.adversary = adversary
        self.gp_model = gp_model
        self.gp_likelihood = gp_likelihood

        return gp_model, gp_likelihood


def train_dptr_gp(source_x: torch.Tensor,
                  source_y: torch.Tensor,
                  target_x: torch.Tensor,
                  target_y: torch.Tensor,
                  latent_dim: int = 10,
                  hidden_dim: int = 32,
                  vae_epochs: int = 100,
                  gp_epochs: int = 50,
                  beta: float = 0.5,
                  lambda_align: float = 1.0,
                  n_mc_samples: int = 1,
                  verbose: bool = False) -> Tuple:
    """
    Train complete DPTR framework: DGRM + MC-DAT + GP.

    CORRECTED implementation following Chai et al. (2022):
    1. PriorNet learns conditional prior p(z|x)
    2. Conditional KL divergence formula
    3. Adversarial alignment (MC-DAT) instead of MMD

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
        lambda_align: Weight for MC-DAT alignment
        n_mc_samples: Monte Carlo samples for MC-DAT
        verbose: Print progress

    Returns:
        vae: Trained DPTRVAE
        gp_model: Trained GP in latent space
        likelihood: GP likelihood
        adversary_net: Trained domain discriminator
        dptr_info: Training information dict
    """
    # ===== Initialize networks =====
    vae = DPTRVAE(
        source_dim=source_x.shape[-1],
        target_dim=target_x.shape[-1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    )

    adversary_net = AdversaryNet(latent_dim=latent_dim)

    if verbose:
        print("="*70)
        print("Training DPTR (Corrected Architecture)")
        print("="*70)
        print(f"Source: {source_x.shape}, Target: {target_x.shape}")
        print(f"Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")
        print(f"VAE epochs: {vae_epochs}, GP epochs: {gp_epochs}")
        print(f"Beta: {beta}, Lambda_align: {lambda_align}")
        print()

    # ===== Train VAE with adversarial alignment =====
    if verbose:
        print("Stage 1: Training DGRM + MC-DAT...")

    vae, adversary_net, _ = train_dptr_vae(
        vae, adversary_net,
        source_x, source_y,
        target_x, target_y,
        n_epochs=vae_epochs,
        beta=beta,
        lambda_align=lambda_align,
        n_mc_samples=n_mc_samples,
        verbose=verbose
    )

    # ===== Encode to latent space =====
    vae.eval()
    with torch.no_grad():
        # Use posterior mean for encoding (more stable than sampling)
        _, mu_s, _ = vae.encode_source(source_x, source_y)
        _, mu_t, _ = vae.encode_target(target_x, target_y)

        z_source = mu_s
        z_target = mu_t

    # ===== Train GP in latent space =====
    if verbose:
        print(f"\nStage 2: Training GP in {latent_dim}D latent space...")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = DPTRGaussianProcess(z_source, source_y, likelihood)

    from src.models.gp_model import train_baseline_gp
    gp_model, likelihood, _ = train_baseline_gp(
        gp_model, likelihood, z_source, source_y,
        num_iter=gp_epochs,
        verbose=verbose
    )

    if verbose:
        print("\n" + "="*70)
        print("DPTR Training Complete!")
        print("="*70)

    dptr_info = {
        'latent_dim': latent_dim,
        'z_source': z_source.detach().cpu().numpy(),
        'z_target': z_target.detach().cpu().numpy(),
        'vae_beta': beta,
        'lambda_align': lambda_align,
    }

    return vae, gp_model, likelihood, adversary_net, dptr_info


def predict_dptr(vae: DPTRVAE,
                gp_model: DPTRGaussianProcess,
                likelihood: gpytorch.likelihoods.Likelihood,
                target_x: torch.Tensor,
                source_domain: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using trained DPTR.

    Online inference uses: x -> PriorNet -> z -> GP -> y
    (Uses prior mean, not posterior, since we don't have y yet)

    Args:
        vae: Trained DPTRVAE
        gp_model: Trained GP
        likelihood: GP likelihood
        target_x: Features to predict [N, D]
        source_domain: If True, use source prior

    Returns:
        predictions: Mean predictions [N]
        uncertainties: Prediction std [N]
    """
    vae.eval()
    gp_model.eval()

    with torch.no_grad():
        # Encode using PRIOR (not posterior, since we don't have y)
        if source_domain:
            mu_prior, _ = vae.prior_source(target_x)
        else:
            mu_prior, _ = vae.prior_target(target_x)

        z = mu_prior  # Use prior mean for inference

        # GP prediction in latent space
        f_dist = gp_model(z)
        pred_dist = likelihood(f_dist)

        predictions = pred_dist.mean.cpu().numpy()
        uncertainties = pred_dist.stddev.cpu().numpy()

    return predictions, uncertainties
