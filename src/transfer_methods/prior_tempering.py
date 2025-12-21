"""
Bayesian Prior Transfer with Tempering

Implements the probabilistic transfer framework where the posterior from the
source domain serves as a tempered prior for the target domain:

    p(θ | D_T, D_S) ∝ p(D_T | θ) · [p(θ | D_S)]^β

where β ∈ [0, 1] is the temperature parameter.

References:
    - Khalil et al. (2021) "Probabilistic Transfer Learning"
    - Pijlman et al. (2025) "Transfer Learning with Prior Tempering"
"""

import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Dict, Optional, Tuple
import numpy as np


class TemperedGP(gpytorch.models.ExactGP):
    """
    Gaussian Process with tempered prior from source domain.

    This implementation allows transfer of knowledge from a source city (e.g., Dublin)
    to a target city by using the source model's posterior as an informative prior,
    weighted by temperature parameter β.

    Parameters
    ----------
    train_x : torch.Tensor
        Target domain training features [N, D]
    train_y : torch.Tensor
        Target domain training observations [N]
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function (typically Gaussian)
    source_hyperparams : Dict[str, torch.Tensor], optional
        Dictionary containing source model hyperparameters:
        - 'lengthscale': Kernel lengthscales
        - 'outputscale': Kernel output scale
        - 'mean_constant': Mean function constant
        - 'noise': Observation noise
    beta : float, default=1.0
        Temperature parameter for prior tempering:
        - β = 1.0: Full trust in source knowledge
        - β = 0.0: Ignore source (standard GP)
        - 0 < β < 1: Partial transfer with tempering

    Attributes
    ----------
    mean_module : gpytorch.means.ConstantMean
        Mean function
    covar_module : gpytorch.kernels.ScaleKernel
        Covariance kernel (RBF with ARD)
    beta : float
        Temperature parameter

    Examples
    --------
    >>> # Train source model on Dublin data
    >>> source_gp = train_source_gp(dublin_x, dublin_y)
    >>>
    >>> # Extract hyperparameters
    >>> source_params = {
    ...     'lengthscale': source_gp.covar_module.base_kernel.lengthscale,
    ...     'outputscale': source_gp.covar_module.outputscale,
    ...     'noise': source_gp.likelihood.noise
    ... }
    >>>
    >>> # Transfer to target city with β=0.5
    >>> target_gp = TemperedGP(
    ...     target_x, target_y, likelihood,
    ...     source_hyperparams=source_params,
    ...     beta=0.5
    ... )
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        source_hyperparams: Optional[Dict[str, torch.Tensor]] = None,
        beta: float = 1.0
    ):
        super().__init__(train_x, train_y, likelihood)

        self.beta = beta
        self.input_dim = train_x.shape[-1]

        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Covariance module with ARD (Automatic Relevance Determination)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)
        )

        # Initialize from source if provided
        if source_hyperparams is not None:
            self._initialize_from_source(source_hyperparams)

    def _initialize_from_source(self, source_hyperparams: Dict[str, torch.Tensor]):
        """
        Initialize model hyperparameters from source domain.

        Parameters
        ----------
        source_hyperparams : Dict[str, torch.Tensor]
            Source model hyperparameters
        """
        with torch.no_grad():
            if 'lengthscale' in source_hyperparams:
                self.covar_module.base_kernel.lengthscale = \
                    source_hyperparams['lengthscale'].clone()

            if 'outputscale' in source_hyperparams:
                self.covar_module.outputscale = \
                    source_hyperparams['outputscale'].clone()

            if 'mean_constant' in source_hyperparams:
                self.mean_module.constant = \
                    source_hyperparams['mean_constant'].clone()

            if 'noise' in source_hyperparams:
                self.likelihood.noise = \
                    source_hyperparams['noise'].clone()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass through the GP.

        Parameters
        ----------
        x : torch.Tensor
            Input locations [N, D]

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Predictive distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TemperedMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    Marginal log likelihood with tempered prior from source domain.

    Computes: log p(D_T | θ) + β · log p(θ | D_S)

    The source prior is modeled as a log-normal distribution centered at the
    source hyperparameter values with configurable variance.

    Parameters
    ----------
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function
    model : TemperedGP
        Tempered GP model
    source_hyperparams : Dict[str, torch.Tensor]
        Source hyperparameters for prior
    beta : float, default=1.0
        Temperature parameter
    prior_variance : float, default=0.1
        Variance of log-normal prior around source values
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: TemperedGP,
        source_hyperparams: Dict[str, torch.Tensor],
        beta: float = 1.0,
        prior_variance: float = 0.1
    ):
        super().__init__(likelihood, model)
        self.beta = beta
        self.source_hyperparams = source_hyperparams
        self.prior_variance = prior_variance

    def forward(
        self,
        function_dist: gpytorch.distributions.MultivariateNormal,
        target: torch.Tensor,
        *params
    ) -> torch.Tensor:
        """
        Compute tempered marginal log likelihood.

        Parameters
        ----------
        function_dist : gpytorch.distributions.MultivariateNormal
            GP function distribution
        target : torch.Tensor
            Target observations

        Returns
        -------
        torch.Tensor
            Tempered marginal log likelihood
        """
        # Standard log likelihood on target data
        log_lik = super().forward(function_dist, target, *params)

        # Add tempered prior if β > 0
        if self.beta > 0:
            log_prior = self._compute_tempered_prior()
            return log_lik + self.beta * log_prior

        return log_lik

    def _compute_tempered_prior(self) -> torch.Tensor:
        """
        Compute log p(θ | D_S) as log-normal around source values.

        Returns
        -------
        torch.Tensor
            Log prior probability
        """
        log_prior = torch.tensor(0.0)

        # Prior on lengthscales
        if 'lengthscale' in self.source_hyperparams:
            current_ls = self.model.covar_module.base_kernel.lengthscale
            source_ls = self.source_hyperparams['lengthscale']
            log_prior += -0.5 * torch.sum(
                (torch.log(current_ls) - torch.log(source_ls))**2
            ) / self.prior_variance

        # Prior on outputscale
        if 'outputscale' in self.source_hyperparams:
            current_os = self.model.covar_module.outputscale
            source_os = self.source_hyperparams['outputscale']
            log_prior += -0.5 * (
                torch.log(current_os) - torch.log(source_os)
            )**2 / self.prior_variance

        # Prior on mean
        if 'mean_constant' in self.source_hyperparams:
            current_mean = self.model.mean_module.constant
            source_mean = self.source_hyperparams['mean_constant']
            log_prior += -0.5 * (current_mean - source_mean)**2 / self.prior_variance

        return log_prior


def train_tempered_gp(
    source_gp: gpytorch.models.ExactGP,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    beta: float = 0.5,
    num_iter: int = 100,
    lr: float = 0.01,
    verbose: bool = True
) -> Tuple[TemperedGP, gpytorch.likelihoods.Likelihood]:
    """
    Train GP on target domain with tempered prior from source.

    This is the main function for applying prior tempering transfer learning.

    Parameters
    ----------
    source_gp : gpytorch.models.ExactGP
        Trained GP from source domain (e.g., Dublin)
    target_x : torch.Tensor
        Target domain features [N, D]
    target_y : torch.Tensor
        Target domain observations [N]
    beta : float, default=0.5
        Temperature parameter (0=no transfer, 1=full transfer)
    num_iter : int, default=100
        Number of training iterations
    lr : float, default=0.01
        Learning rate
    verbose : bool, default=True
        Print training progress

    Returns
    -------
    model : TemperedGP
        Trained target model
    likelihood : gpytorch.likelihoods.Likelihood
        Trained likelihood

    Examples
    --------
    >>> # Transfer from Dublin to Cork with β=0.5
    >>> target_model, likelihood = train_tempered_gp(
    ...     source_gp=dublin_gp,
    ...     target_x=cork_x,
    ...     target_y=cork_y,
    ...     beta=0.5,
    ...     num_iter=200
    ... )
    """
    # Extract source hyperparameters
    source_hyperparams = {
        'lengthscale': source_gp.covar_module.base_kernel.lengthscale.detach().clone(),
        'outputscale': source_gp.covar_module.outputscale.detach().clone(),
        'mean_constant': source_gp.mean_module.constant.detach().clone(),
        'noise': source_gp.likelihood.noise.detach().clone()
    }

    # Initialize target model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    target_model = TemperedGP(
        target_x, target_y, likelihood,
        source_hyperparams=source_hyperparams,
        beta=beta
    )

    # Training mode
    target_model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)

    # Tempered MLL
    mll = TemperedMarginalLogLikelihood(
        likelihood, target_model,
        source_hyperparams, beta=beta
    )

    # Training loop
    losses = []
    for i in range(num_iter):
        optimizer.zero_grad()
        output = target_model(target_x)
        loss = -mll(output, target_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (i + 1) % 10 == 0:
            print(f'Iter {i+1}/{num_iter} - Loss: {loss.item():.3f}')

    # Set to eval mode
    target_model.eval()
    likelihood.eval()

    return target_model, likelihood


def extract_hyperparameters(
    model: gpytorch.models.ExactGP
) -> Dict[str, torch.Tensor]:
    """
    Extract hyperparameters from trained GP model.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        Trained GP model

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary of hyperparameters
    """
    return {
        'lengthscale': model.covar_module.base_kernel.lengthscale.detach().clone(),
        'outputscale': model.covar_module.outputscale.detach().clone(),
        'mean_constant': model.mean_module.constant.detach().clone(),
        'noise': model.likelihood.noise.detach().clone()
    }
