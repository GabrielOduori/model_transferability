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
        self.source_hyperparams = source_hyperparams
        self.likelihood_obj = likelihood

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

            if 'noise' in source_hyperparams and hasattr(self, "likelihood_obj"):
                self.likelihood_obj.noise = \
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
    prior_variances : Dict[str, float] or float, optional
        Variance of log-normal prior around source values.
        Can be a single float (applied to all parameters) or a dict with
        parameter-specific variances:
        - 'lengthscale': variance for lengthscale parameters
        - 'outputscale': variance for output scale
        - 'mean_constant': variance for mean constant
        - 'noise': variance for noise parameter
        Default: {'lengthscale': 0.2, 'outputscale': 0.1,
                  'mean_constant': 0.15, 'noise': 0.05}
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: TemperedGP,
        source_hyperparams: Optional[Dict[str, torch.Tensor]] = None,
        beta: float = 1.0,
        prior_variances: Optional[Dict[str, float]] = None
    ):
        super().__init__(likelihood, model)
        self.beta = beta
        if source_hyperparams is None and hasattr(model, "source_hyperparams"):
            source_hyperparams = model.source_hyperparams
        self.source_hyperparams = source_hyperparams
        self.prior_variance = prior_variances if prior_variances is not None else 0.1

        # Set parameter-specific variances
        if prior_variances is None:
            # Default: parameter-specific variances based on typical uncertainty
            self.prior_variances = {
                'lengthscale': 0.2,      # Higher uncertainty in lengthscales
                'outputscale': 0.1,      # Moderate uncertainty
                'mean_constant': 0.15,   # Moderate uncertainty
                'noise': 0.05            # Lower uncertainty in noise
            }
        elif isinstance(prior_variances, dict):
            self.prior_variances = prior_variances
        else:
            # Single value provided - use for all parameters
            self.prior_variances = {
                'lengthscale': prior_variances,
                'outputscale': prior_variances,
                'mean_constant': prior_variances,
                'noise': prior_variances
            }

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
        if self.beta > 0 and self.source_hyperparams is not None:
            log_prior = self._compute_tempered_prior()
            return log_lik + self.beta * log_prior

        return log_lik

    def _compute_tempered_prior(self) -> torch.Tensor:
        """
        Compute log p(θ | D_S) as log-normal around source values.

        Uses parameter-specific variances to better reflect uncertainty
        in different hyperparameters. For example, lengthscales typically
        have higher uncertainty than noise parameters.

        Returns
        -------
        torch.Tensor
            Log prior probability
        """
        if self.source_hyperparams is None:
            return torch.tensor(0.0, device=self.model.train_inputs[0].device)

        device = self.model.train_inputs[0].device
        log_prior = torch.tensor(0.0, device=device)
        eps = 1e-8

        # Prior on lengthscales (log-normal)
        if 'lengthscale' in self.source_hyperparams:
            current_ls = self.model.covar_module.base_kernel.lengthscale
            source_ls = self.source_hyperparams['lengthscale']
            var_ls = self.prior_variances.get('lengthscale', 0.2)
            term = torch.sum((torch.log(current_ls) - torch.log(source_ls))**2) / (var_ls + eps)
            log_prior += -0.5 * term - 0.5 * torch.log(torch.tensor(var_ls + eps, device=device))

        # Prior on outputscale (log-normal)
        if 'outputscale' in self.source_hyperparams:
            current_os = self.model.covar_module.outputscale
            source_os = self.source_hyperparams['outputscale']
            var_os = self.prior_variances.get('outputscale', 0.1)
            term = (torch.log(current_os) - torch.log(source_os))**2 / (var_os + eps)
            log_prior += -0.5 * term - 0.5 * torch.log(torch.tensor(var_os + eps, device=device))

        # Prior on mean (Gaussian)
        if 'mean_constant' in self.source_hyperparams:
            current_mean = self.model.mean_module.constant
            source_mean = self.source_hyperparams['mean_constant']
            var_mean = self.prior_variances.get('mean_constant', 0.15)
            term = (current_mean - source_mean)**2 / (var_mean + eps)
            log_prior += -0.5 * term - 0.5 * torch.log(torch.tensor(var_mean + eps, device=device))

        # Prior on noise (log-normal) - if present
        if 'noise' in self.source_hyperparams and hasattr(self.model, "likelihood_obj"):
            current_noise = self.model.likelihood_obj.noise
            source_noise = self.source_hyperparams['noise']
            var_noise = self.prior_variances.get('noise', 0.05)
            term = (torch.log(current_noise) - torch.log(source_noise))**2 / (var_noise + eps)
            log_prior += -0.5 * torch.sum(term) - 0.5 * torch.log(torch.tensor(var_noise + eps, device=device))

        return log_prior


def train_tempered_gp(
    model: TemperedGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    beta: float = 0.5,
    num_iter: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    prior_variances: Optional[Dict[str, float]] = None
) -> Tuple[TemperedGP, gpytorch.likelihoods.Likelihood, list]:
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
    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tempered MLL
    mll = TemperedMarginalLogLikelihood(
        likelihood, model,
        model.source_hyperparams,
        beta=beta,
        prior_variances=prior_variances
    )

    # Training loop
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

    # Set to eval mode
    model.eval()
    likelihood.eval()

    return model, likelihood, losses


def transfer_with_tempering(
    source_gp: gpytorch.models.ExactGP,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    beta: float = 0.5,
    num_iter: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
    prior_variances: Optional[Dict[str, float]] = None
) -> Tuple[TemperedGP, gpytorch.likelihoods.Likelihood]:
    """
    Simple wrapper for training GP with tempered prior from source.

    This provides a simpler API that matches the documentation.

    Parameters
    ----------
    source_gp : gpytorch.models.ExactGP
        Trained GP from source domain
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
    verbose : bool, default=False
        Print training progress
    prior_variances : Optional[Dict[str, float]], default=None
        Parameter-specific prior variances

    Returns
    -------
    model : TemperedGP
        Trained target model
    likelihood : gpytorch.likelihoods.Likelihood
        Trained likelihood

    Examples
    --------
    >>> # Transfer from Dublin to Cork
    >>> target_model, likelihood = transfer_with_tempering(
    ...     source_gp=dublin_gp,
    ...     target_x=cork_x,
    ...     target_y=cork_y,
    ...     beta=0.5
    ... )
    """
    # Extract source hyperparameters
    source_hyperparams = extract_hyperparameters(source_gp)

    # Create TemperedGP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = TemperedGP(
        train_x=target_x,
        train_y=target_y,
        likelihood=likelihood,
        source_hyperparams=source_hyperparams,
        beta=beta
    )

    # Train
    model, likelihood, _ = train_tempered_gp(
        model, likelihood, target_x, target_y,
        beta=beta,
        num_iter=num_iter,
        lr=lr,
        verbose=verbose,
        prior_variances=prior_variances
    )

    return model, likelihood


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
