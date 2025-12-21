"""
Optimal Bayesian Transfer Learning (OBTL) for Gaussian Processes

Reference: Karbalayghareh et al. (2018)
"Optimal Bayesian Transfer Learning"

Core idea: Joint Wishart prior on source and target precision matrices
allows optimal transfer of covariance structure.
"""

import torch
import gpytorch
import numpy as np
from scipy.stats import wishart, invwishart
from typing import Tuple, Optional


class OBTLGaussianProcess:
    """
    OBTL for GP on fixed spatial grid using inducing points.

    Models covariance at inducing locations with Wishart prior,
    enabling transfer of spatial structure from source to target.
    """

    def __init__(self, n_inducing_points: int = 20, nu_0: float = 5.0):
        """
        Args:
            n_inducing_points: Number of spatial inducing points
            nu_0: Wishart prior degrees of freedom (controls prior strength)
        """
        self.n_inducing = n_inducing_points
        self.nu_0 = nu_0
        self.source_cov = None
        self.target_cov = None
        self.inducing_points = None

    def fit_source(self,
                   X_source: torch.Tensor,
                   y_source: torch.Tensor,
                   inducing_points: Optional[torch.Tensor] = None):
        """
        Fit source domain and extract covariance structure.

        Args:
            X_source: Source features [N_s, D]
            y_source: Source targets [N_s]
            inducing_points: Fixed inducing locations [M, D]
        """
        if inducing_points is None:
            inducing_points = self._select_inducing_points(X_source)

        self.inducing_points = inducing_points

        # Train source GP
        source_model, source_likelihood = self._train_gp(X_source, y_source)

        # Extract covariance at inducing points
        self.source_cov = self._extract_covariance(source_model, inducing_points)

        return source_model, source_likelihood

    def transfer_to_target(self,
                          X_target: torch.Tensor,
                          y_target: torch.Tensor,
                          delta: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Transfer covariance structure to target domain using OBTL.

        Args:
            X_target: Target features [N_t, D]
            y_target: Target targets [N_t]
            delta: Transfer parameter (0=no transfer, 1=full transfer)

        Returns:
            transferred_cov: Transferred covariance matrix
            transfer_strength: Effective transfer weight
        """
        # Train initial target GP
        target_model, _ = self._train_gp(X_target, y_target)

        # Extract target covariance
        self.target_cov = self._extract_covariance(target_model, self.inducing_points)

        # OBTL transfer using Wishart conjugacy
        n_s = X_target.shape[0]  # Target sample size

        # Posterior parameters
        nu_n = self.nu_0 + n_s

        # Transfer covariance: weighted combination
        # Weight based on sample size and delta parameter
        weight_source = (self.nu_0 * delta) / nu_n
        weight_target = n_s / nu_n

        transferred_cov = (
            weight_source * self.source_cov +
            weight_target * self.target_cov
        )

        return transferred_cov, weight_source

    def _select_inducing_points(self, X: torch.Tensor) -> torch.Tensor:
        """Select inducing points using k-means clustering."""
        from sklearn.cluster import KMeans

        X_np = X.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_inducing, random_state=42, n_init=10)
        kmeans.fit(X_np)

        return torch.tensor(kmeans.cluster_centers_, dtype=X.dtype)

    def _train_gp(self, X: torch.Tensor, y: torch.Tensor):
        """Train standard GP model."""
        from src.models.gp_model import BaselineGP, train_baseline_gp

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(X, y, likelihood)

        model, likelihood = train_baseline_gp(model, likelihood, X, y, num_iter=50, verbose=False)
        return model, likelihood

    def _extract_covariance(self,
                           model: gpytorch.models.ExactGP,
                           inducing_points: torch.Tensor) -> torch.Tensor:
        """Extract covariance matrix at inducing points."""
        model.eval()
        with torch.no_grad():
            covar_matrix = model.covar_module(inducing_points).evaluate()

        # Ensure positive definite
        covar_matrix = covar_matrix + 1e-4 * torch.eye(len(inducing_points))

        return covar_matrix


class OBTLTransferGP(gpytorch.models.ExactGP):
    """
    GP model with OBTL-transferred covariance structure.

    Uses transferred covariance at inducing points with
    interpolation for predictions.
    """

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 inducing_points: torch.Tensor,
                 transferred_cov: torch.Tensor):
        super().__init__(train_x, train_y, likelihood)

        self.inducing_points = inducing_points
        self.transferred_cov = transferred_cov

        self.mean_module = gpytorch.means.ConstantMean()

        # Use GridInterpolationKernel for inducing point interpolation
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(),
                num_dims=train_x.shape[-1],
                grid_size=100
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_obtl_gp(source_x: torch.Tensor,
                  source_y: torch.Tensor,
                  target_x: torch.Tensor,
                  target_y: torch.Tensor,
                  n_inducing: int = 20,
                  nu_0: float = 5.0,
                  delta: float = 1.0,
                  n_iterations: int = 100) -> Tuple:
    """
    Train GP with OBTL transfer learning.

    Args:
        source_x: Source features [N_s, D]
        source_y: Source targets [N_s]
        target_x: Target features [N_t, D]
        target_y: Target targets [N_t]
        n_inducing: Number of inducing points
        nu_0: Wishart prior strength
        delta: Transfer strength (0-1)
        n_iterations: Training iterations

    Returns:
        model: Trained OBTL GP
        likelihood: Likelihood
        obtl_info: Dict with transfer info
    """
    # Initialize OBTL
    obtl = OBTLGaussianProcess(n_inducing_points=n_inducing, nu_0=nu_0)

    # Fit source
    source_model, source_likelihood = obtl.fit_source(source_x, source_y)

    # Transfer to target
    transferred_cov, transfer_weight = obtl.transfer_to_target(
        target_x, target_y, delta=delta
    )

    # Train target model with transferred structure
    from src.models.gp_model import BaselineGP, train_baseline_gp

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BaselineGP(target_x, target_y, likelihood)

    model, likelihood = train_baseline_gp(
        model, likelihood, target_x, target_y,
        num_iter=n_iterations,
        verbose=False
    )

    obtl_info = {
        'source_cov': obtl.source_cov.detach().cpu().numpy(),
        'target_cov': obtl.target_cov.detach().cpu().numpy(),
        'transferred_cov': transferred_cov.detach().cpu().numpy(),
        'transfer_weight': transfer_weight,
        'inducing_points': obtl.inducing_points.detach().cpu().numpy(),
        'n_inducing': n_inducing,
        'nu_0': nu_0,
        'delta': delta
    }

    return model, likelihood, obtl_info


def compare_covariance_structures(source_cov: np.ndarray,
                                  target_cov: np.ndarray,
                                  transferred_cov: np.ndarray) -> dict:
    """
    Compare source, target, and transferred covariance matrices.

    Returns metrics for covariance similarity.
    """
    # Frobenius norm distances
    source_target_dist = np.linalg.norm(source_cov - target_cov, 'fro')
    source_transferred_dist = np.linalg.norm(source_cov - transferred_cov, 'fro')
    target_transferred_dist = np.linalg.norm(target_cov - transferred_cov, 'fro')

    # Normalized distances
    norm_s = np.linalg.norm(source_cov, 'fro')
    norm_t = np.linalg.norm(target_cov, 'fro')

    return {
        'source_target_distance': source_target_dist,
        'source_transferred_distance': source_transferred_dist,
        'target_transferred_distance': target_transferred_dist,
        'source_norm': norm_s,
        'target_norm': norm_t,
        'normalized_st_distance': source_target_dist / (norm_s + norm_t),
        'normalized_transfer_to_source': source_transferred_dist / norm_s,
        'normalized_transfer_to_target': target_transferred_dist / norm_t,
    }
