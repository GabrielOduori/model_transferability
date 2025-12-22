"""
Transfer Learning for FusionSVGP Models

Implements transfer strategies for multi-source Sparse Variational GPs:
1. Kernel hyperparameter transfer (spatial/temporal lengthscales, outputscale)
2. Inducing point initialization from source model
3. Likelihood parameter transfer (noise variances, calibration)
4. Complete hybrid transfer pipeline

Requires the fusiongp package installed.
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import sys
import os

# Add fusiongp to path - check environment variable or use default
fusiongp_root = Path(os.environ.get(
    "FUSIONGP_PATH",
    "/media/gabriel-oduori/SERVER/dev_space/fusiongp2/fusiongp"
))

# Use importlib to load fusiongp modules dynamically to avoid namespace conflicts
# with this project's 'src' package
try:
    if not fusiongp_root.exists():
        raise ImportError(f"fusiongp root directory not found at {fusiongp_root}")

    # Temporarily manipulate sys.path and sys.modules to import from fusiongp
    import importlib.util

    # Load modules directly from fusiongp
    svgp_path = fusiongp_root / "src" / "models" / "svgp.py"
    kernels_path = fusiongp_root / "src" / "models" / "kernels.py"
    likelihoods_path = fusiongp_root / "src" / "models" / "likelihoods.py"

    if not all(p.exists() for p in [svgp_path, kernels_path, likelihoods_path]):
        raise ImportError("fusiongp model files not found")

    # Add fusiongp to path temporarily for internal imports within fusiongp modules
    sys.path.insert(0, str(fusiongp_root))

    # Load kernels module
    spec_kernels = importlib.util.spec_from_file_location("fusiongp_kernels", kernels_path)
    fusiongp_kernels = importlib.util.module_from_spec(spec_kernels)
    sys.modules["src.models.kernels"] = fusiongp_kernels  # Make it available for svgp's imports
    spec_kernels.loader.exec_module(fusiongp_kernels)

    # Load likelihoods module
    spec_likelihoods = importlib.util.spec_from_file_location("fusiongp_likelihoods", likelihoods_path)
    fusiongp_likelihoods = importlib.util.module_from_spec(spec_likelihoods)
    sys.modules["src.models.likelihoods"] = fusiongp_likelihoods  # Make it available for svgp's imports
    spec_likelihoods.loader.exec_module(fusiongp_likelihoods)

    # Load svgp module
    spec_svgp = importlib.util.spec_from_file_location("fusiongp_svgp", svgp_path)
    fusiongp_svgp = importlib.util.module_from_spec(spec_svgp)
    spec_svgp.loader.exec_module(fusiongp_svgp)

    # Extract the classes we need
    FusionSVGP = fusiongp_svgp.FusionSVGP
    SpatioTemporalKernel = fusiongp_kernels.SpatioTemporalKernel
    MultiSourceLikelihood = fusiongp_likelihoods.MultiSourceLikelihood

    FUSION_GP_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError) as e:
    FUSION_GP_AVAILABLE = False
    print("Warning: fusiongp not available. Install from github.com/GabrielOduori/fusionGP2")
    print(f"   Import error details: {e}")


def transfer_kernel_hyperparameters(
    source_model: 'FusionSVGP',
    target_model: 'FusionSVGP',
    alpha: float = 0.5
) -> 'FusionSVGP':
    """
    Transfer kernel hyperparameters from source to target with weighted averaging.

    Args:
        source_model: Fitted source FusionSVGP
        target_model: Target FusionSVGP (may be fitted or unfitted)
        alpha: Weight for source hyperparameters (0=target only, 1=source only)

    Returns:
        Target model with transferred hyperparameters
    """
    if not FUSION_GP_AVAILABLE:
        raise ImportError("fusiongp package required")

    # Transfer spatial lengthscales
    source_spatial_ls = source_model.covar_module.spatial_lengthscale.detach()
    target_spatial_ls = target_model.covar_module.spatial_lengthscale.detach()

    transferred_spatial_ls = alpha * source_spatial_ls + (1 - alpha) * target_spatial_ls
    target_model.covar_module.spatial_kernel.lengthscale = transferred_spatial_ls

    # Transfer temporal lengthscale
    source_temporal_ls = source_model.covar_module.temporal_lengthscale.detach()
    target_temporal_ls = target_model.covar_module.temporal_lengthscale.detach()

    transferred_temporal_ls = alpha * source_temporal_ls + (1 - alpha) * target_temporal_ls
    target_model.covar_module.temporal_kernel.lengthscale = transferred_temporal_ls

    # Transfer outputscale
    source_outputscale = source_model.covar_module.outputscale.detach()
    target_outputscale = target_model.covar_module.outputscale.detach()

    transferred_outputscale = alpha * source_outputscale + (1 - alpha) * target_outputscale
    target_model.covar_module.outputscale_param.data = transferred_outputscale

    return target_model


def transfer_inducing_points(
    source_model: 'FusionSVGP',
    target_model: 'FusionSVGP',
    target_x: torch.Tensor,
    method: str = 'hybrid',
    beta: float = 0.3
) -> 'FusionSVGP':
    """
    Initialize target inducing points using source model locations.

    Args:
        source_model: Fitted source model
        target_model: Target model
        target_x: Target training data for k-means initialization
        method: 'source' (use source directly), 'kmeans' (target k-means),
                'hybrid' (weighted combination)
        beta: Weight for source inducing points in hybrid mode

    Returns:
        Target model with initialized inducing points
    """
    if not FUSION_GP_AVAILABLE:
        raise ImportError("fusiongp package required")

    source_inducing = source_model.get_inducing_points()

    if method == 'source':
        # Use source inducing points directly
        target_model.variational_strategy.inducing_points.data = source_inducing.clone()

    elif method == 'kmeans':
        # Initialize from target data using k-means
        target_model.initialize_inducing_points(target_x, method='kmeans')

    elif method == 'hybrid':
        # Weighted combination of source inducing points and target k-means
        target_model.initialize_inducing_points(target_x, method='kmeans')
        target_inducing = target_model.get_inducing_points()

        # Ensure compatible shapes
        n_source = source_inducing.shape[0]
        n_target = target_inducing.shape[0]

        if n_source == n_target:
            # Simple weighted average
            transferred_inducing = beta * source_inducing + (1 - beta) * target_inducing
            target_model.variational_strategy.inducing_points.data = transferred_inducing
        else:
            # Use source as initialization hint but keep target k-means
            # (Different numbers of inducing points - can't directly combine)
            pass

    else:
        raise ValueError(f"Unknown method: {method}")

    return target_model


def transfer_likelihood_parameters(
    source_model: 'FusionSVGP',
    target_model: 'FusionSVGP',
    gamma: float = 0.5,
    transfer_calibration: bool = True
) -> 'FusionSVGP':
    """
    Transfer likelihood parameters (noise variances, calibration).

    Args:
        source_model: Fitted source model
        target_model: Target model
        gamma: Weight for source likelihood parameters
        transfer_calibration: Whether to transfer low-cost calibration params

    Returns:
        Target model with transferred likelihood parameters
    """
    if not FUSION_GP_AVAILABLE:
        raise ImportError("fusiongp package required")

    # Transfer noise parameters for each source
    for source_name in target_model.likelihood.sources:
        if source_name in source_model.likelihood.sources:
            source_raw_noise = source_model.likelihood.raw_noise[source_name].detach()
            target_raw_noise = target_model.likelihood.raw_noise[source_name].detach()

            transferred_raw_noise = gamma * source_raw_noise + (1 - gamma) * target_raw_noise
            target_model.likelihood.raw_noise[source_name].data = transferred_raw_noise

    # Transfer calibration parameters
    if transfer_calibration:
        source_slope = source_model.likelihood.raw_lc_slope.detach()
        target_slope = target_model.likelihood.raw_lc_slope.detach()

        transferred_slope = gamma * source_slope + (1 - gamma) * target_slope
        target_model.likelihood.raw_lc_slope.data = transferred_slope

        source_intercept = source_model.likelihood.raw_lc_intercept.detach()
        target_intercept = target_model.likelihood.raw_lc_intercept.detach()

        transferred_intercept = gamma * source_intercept + (1 - gamma) * target_intercept
        target_model.likelihood.raw_lc_intercept.data = transferred_intercept

    return target_model


def hybrid_fusion_transfer(
    source_model: 'FusionSVGP',
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_source_masks: torch.Tensor,
    kernel_weight: float = 0.5,
    inducing_weight: float = 0.3,
    likelihood_weight: float = 0.5,
    n_inducing: Optional[int] = None,
    fine_tune: bool = True,
    fine_tune_epochs: int = 50
) -> Tuple['FusionSVGP', Dict]:
    """
    Complete transfer learning pipeline for FusionSVGP.

    Transfers:
    1. Kernel hyperparameters (spatial/temporal lengthscales, outputscale)
    2. Inducing point locations
    3. Likelihood parameters (noise, calibration)

    Args:
        source_model: Fitted source FusionSVGP
        target_x: Target input locations (N, 3)
        target_y: Target observations (N, n_sources)
        target_source_masks: Target observation masks (N, n_sources)
        kernel_weight: Weight for source kernel hyperparameters
        inducing_weight: Weight for source inducing points
        likelihood_weight: Weight for source likelihood parameters
        n_inducing: Number of inducing points for target (default: same as source)
        fine_tune: Whether to fine-tune after transfer
        fine_tune_epochs: Fine-tuning epochs

    Returns:
        target_model: Model with transferred parameters
        transfer_info: Dictionary with transfer statistics
    """
    if not FUSION_GP_AVAILABLE:
        raise ImportError("fusiongp package required")

    # Get source configuration
    n_inducing = n_inducing or source_model.n_inducing

    # Create target model with same architecture
    target_model = FusionSVGP(
        n_inducing=n_inducing,
        kernel_type=source_model.kernel_type,
        sources=source_model.sources,
        learn_inducing_locations=source_model.learn_inducing_locations
    )

    # Step 1: Initialize target model on target data
    target_model.initialize_inducing_points(target_x, method='kmeans')

    # Perform initial training to get target hyperparameters
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    target_model.train()

    for _ in range(20):  # Quick initialization
        optimizer.zero_grad()
        loss = -target_model.elbo(target_x, target_y, target_source_masks)
        loss.backward()
        optimizer.step()

    # Step 2: Transfer kernel hyperparameters
    target_model = transfer_kernel_hyperparameters(
        source_model, target_model, alpha=kernel_weight
    )

    # Step 3: Transfer inducing points
    target_model = transfer_inducing_points(
        source_model, target_model, target_x,
        method='hybrid', beta=inducing_weight
    )

    # Step 4: Transfer likelihood parameters
    target_model = transfer_likelihood_parameters(
        source_model, target_model, gamma=likelihood_weight
    )

    # Step 5: Fine-tune (optional)
    if fine_tune:
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.005)
        target_model.train()

        for _ in range(fine_tune_epochs):
            optimizer.zero_grad()
            loss = -target_model.elbo(target_x, target_y, target_source_masks)
            loss.backward()
            optimizer.step()

    # Collect transfer info
    transfer_info = {
        'kernel_weight': kernel_weight,
        'inducing_weight': inducing_weight,
        'likelihood_weight': likelihood_weight,
        'n_inducing': n_inducing,
        'fine_tuned': fine_tune,
        'source_hyperparams': source_model.get_hyperparameters(),
        'target_hyperparams': target_model.get_hyperparameters(),
    }

    return target_model, transfer_info


class TransferableFusionSVGP:
    """
    Wrapper for FusionSVGP with transfer learning capabilities.

    Simplifies transfer learning workflow.
    """

    def __init__(self, base_model: Optional['FusionSVGP'] = None, **model_kwargs):
        """
        Args:
            base_model: Existing model or None to create new
            **model_kwargs: Parameters for FusionSVGP
        """
        if not FUSION_GP_AVAILABLE:
            raise ImportError("fusiongp package required")

        if base_model is not None:
            self.model = base_model
        else:
            self.model = FusionSVGP(**model_kwargs)

        self.transfer_history = []

    def initialize_inducing_points(self, x, method='kmeans'):
        """Initialize inducing points."""
        self.model.initialize_inducing_points(x, method=method)
        return self

    def train(self, train_x, train_y, source_masks, epochs=100, lr=0.01):
        """Train the model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = -self.model.elbo(train_x, train_y, source_masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def transfer_from(
        self,
        source: 'TransferableFusionSVGP',
        target_x, target_y, target_source_masks,
        kernel_weight=0.5,
        inducing_weight=0.3,
        likelihood_weight=0.5
    ):
        """
        Transfer from source model.

        Args:
            source: Source TransferableFusionSVGP
            target_x, target_y, target_source_masks: Target data
            kernel_weight: Kernel hyperparameter transfer weight
            inducing_weight: Inducing point transfer weight
            likelihood_weight: Likelihood parameter transfer weight

        Returns:
            self with transferred parameters
        """
        self.model, transfer_info = hybrid_fusion_transfer(
            source.model, target_x, target_y, target_source_masks,
            kernel_weight, inducing_weight, likelihood_weight
        )

        self.transfer_history.append({
            'source': source,
            'transfer_info': transfer_info
        })

        return self

    def predict(self, x, include_noise=False, source='epa'):
        """Make predictions."""
        return self.model.predict(x, include_noise=include_noise, source=source)

    def get_hyperparameters(self):
        """Get model hyperparameters."""
        return self.model.get_hyperparameters()


def get_transfer_summary(source_model: 'FusionSVGP',
                        target_model: 'FusionSVGP') -> Dict:
    """
    Summarize differences between source and target models.

    Returns:
        Dictionary with comparison metrics
    """
    summary = {}

    # Kernel hyperparameter differences
    source_params = source_model.get_hyperparameters()
    target_params = target_model.get_hyperparameters()

    # Spatial lengthscale difference
    spatial_diff = np.linalg.norm(
        source_params['spatial_lengthscale'] - target_params['spatial_lengthscale']
    )
    summary['spatial_lengthscale_diff'] = spatial_diff

    # Temporal lengthscale difference
    temporal_diff = abs(
        source_params['temporal_lengthscale'] - target_params['temporal_lengthscale']
    )
    summary['temporal_lengthscale_diff'] = float(temporal_diff)

    # Outputscale difference
    outputscale_diff = abs(
        source_params['outputscale'] - target_params['outputscale']
    )
    summary['outputscale_diff'] = float(outputscale_diff)

    # Inducing point differences
    source_inducing = source_model.get_inducing_points()
    target_inducing = target_model.get_inducing_points()

    if source_inducing.shape == target_inducing.shape:
        inducing_diff = torch.norm(source_inducing - target_inducing).item()
        summary['inducing_points_diff'] = inducing_diff

    return summary
