"""
Baseline Gaussian Process Model for Air Quality Prediction

This module provides standard GP implementations without transfer learning,
serving as baselines for comparison.
"""

import torch
import gpytorch
from typing import Tuple, Optional
import numpy as np
import copy


class BaselineGP(gpytorch.models.ExactGP):
    """
    Standard Gaussian Process for air quality prediction.

    This serves as the baseline (no transfer) comparison for transfer
    learning experiments.

    Parameters
    ----------
    train_x : torch.Tensor
        Training features [N, D]
    train_y : torch.Tensor
        Training observations [N]
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function
    ard : bool, default=True
        Use Automatic Relevance Determination (different lengthscale per dimension)

    Examples
    --------
    >>> # Train baseline GP on target city from scratch
    >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = BaselineGP(train_x, train_y, likelihood)
    >>> trained_model = train_baseline_gp(model, likelihood, train_x, train_y)
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        ard: bool = True
    ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_baseline_gp(
    model: BaselineGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_iter: int = 100,
    lr: float = 0.1,
    verbose: bool = True
) -> Tuple[BaselineGP, gpytorch.likelihoods.Likelihood, list]:
    """
    Train baseline GP model.

    Parameters
    ----------
    model : BaselineGP
        Model to train
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function
    train_x : torch.Tensor
        Training features
    train_y : torch.Tensor
        Training observations
    num_iter : int, default=100
        Training iterations
    lr : float, default=0.1
        Learning rate
    verbose : bool, default=True
        Print progress

    Returns
    -------
    model : BaselineGP
        Trained model
    likelihood : gpytorch.likelihoods.Likelihood
        Trained likelihood
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (i + 1) % 10 == 0:
            print(f'Iter {i+1}/{num_iter} - Loss: {loss.item():.3f}')

    model.eval()
    likelihood.eval()

    return model, likelihood, losses


def predict_with_uncertainty(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    test_x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions with uncertainty quantification.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        Trained GP model
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function
    test_x : torch.Tensor
        Test inputs [N_test, D]

    Returns
    -------
    predictions : np.ndarray
        Mean predictions [N_test]
    uncertainties : np.ndarray
        Standard deviations [N_test]
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        predictions = pred_dist.mean.squeeze(-1)
        uncertainties = pred_dist.stddev.squeeze(-1)

    return predictions, uncertainties


def sample_posterior(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_samples: int = 1000,
    prior_variance: float = 0.1,
    sample_lengthscale: bool = True,
    sample_outputscale: bool = True,
    sample_noise: bool = True
) -> list:
    """
    Sample from posterior distribution of hyperparameters.

    This is used for KL divergence computation in RQ1.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        Trained GP model
    n_samples : int, default=1000
        Number of posterior samples
    sample_lengthscale : bool, default=True
        Include lengthscale in samples
    sample_outputscale : bool, default=True
        Include outputscale in samples
    sample_noise : bool, default=True
        Include noise in samples

    Returns
    -------
    np.ndarray
        Posterior samples [n_samples, D]
        where D = dim(lengthscale) + dim(outputscale) + dim(noise)

    Notes
    -----
    This is a simplified implementation that assumes Gaussian approximation
    around the MAP estimate. For more rigorous sampling, use MCMC or
    variational inference.
    """
    # Simple Gaussian approximation around MAP estimate
    sampled_models = []

    base_lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu()
    base_outputscale = model.covar_module.outputscale.detach().cpu()
    base_noise = likelihood.noise.detach().cpu()

    scale = float(np.sqrt(max(prior_variance, 1e-6)))

    for _ in range(n_samples):
        m_copy = copy.deepcopy(model)

        with torch.no_grad():
            if sample_lengthscale:
                noise_factor = torch.exp(torch.randn_like(base_lengthscale) * scale)
                m_copy.covar_module.base_kernel.lengthscale = base_lengthscale * noise_factor

            if sample_outputscale:
                os_factor = torch.exp(torch.randn_like(base_outputscale) * scale)
                m_copy.covar_module.outputscale = base_outputscale * os_factor

            if sample_noise and hasattr(likelihood, "noise"):
                noise_factor = torch.exp(torch.randn_like(base_noise) * scale)
                if hasattr(m_copy, "likelihood"):
                    m_copy.likelihood.noise = base_noise * noise_factor

        sampled_models.append(m_copy)

    return sampled_models


class SpatialTemporalGP(gpytorch.models.ExactGP):
    """
    GP with separate spatial and temporal kernels.

    k(x, x') = k_spatial(s, s') × k_temporal(t, t')

    This is more appropriate for air quality data with explicit
    spatial-temporal structure.

    Parameters
    ----------
    train_x : torch.Tensor
        Training features [N, D_spatial + D_temporal]
    train_y : torch.Tensor
        Training observations [N]
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function
    spatial_dims : list of int
        Indices of spatial dimensions
    temporal_dims : list of int
        Indices of temporal dimensions
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        spatial_dims: list,
        temporal_dims: list
    ):
        super().__init__(train_x, train_y, likelihood)

        self.spatial_dims = spatial_dims
        self.temporal_dims = temporal_dims

        self.mean_module = gpytorch.means.ConstantMean()

        # Spatial kernel (RBF)
        spatial_kernel = gpytorch.kernels.RBFKernel(
            active_dims=spatial_dims,
            ard_num_dims=len(spatial_dims)
        )

        # Temporal kernel (Matern with nu=3/2 for smoothness)
        temporal_kernel = gpytorch.kernels.MaternKernel(
            nu=1.5,
            active_dims=temporal_dims,
            ard_num_dims=len(temporal_dims)
        )

        # Product kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            spatial_kernel * temporal_kernel
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
