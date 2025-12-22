"""
Transfer Learning for GAM-SSM-LUR Models

Implements transfer strategies for Hybrid GAM-SSM-LUR models:
1. LUR coefficient transfer (spatial component)
2. State space dynamics transfer (temporal component)
3. Combined hybrid transfer

Requires the gam_ssm_lur package installed.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
import sys

# Add gam_ssm_lur to path if available
gam_ssm_lur_path = Path("/media/gabriel-oduori/SERVER/dev_space/gam_ssm_lur/src")
if gam_ssm_lur_path.exists():
    sys.path.insert(0, str(gam_ssm_lur_path))

try:
    from gam_ssm_lur.models.hybrid import HybridGAMSSM, HybridPrediction
    from gam_ssm_lur.models.spatial_gam import SpatialGAM
    from gam_ssm_lur.models.state_space import StateSpaceModel

    # Test if pygam is available by trying to import it
    try:
        import pygam
        GAM_SSM_AVAILABLE = True
    except ImportError:
        GAM_SSM_AVAILABLE = False
        print("Warning: pygam not available. Install with: pip install pygam")

except ImportError:
    GAM_SSM_AVAILABLE = False
    print("Warning: gam_ssm_lur not available. Install from github.com/GabrielOduori/gam_ssm_lur")


def transfer_lur_coefficients(source_model: 'HybridGAMSSM',
                              target_X: np.ndarray,
                              target_y: np.ndarray,
                              target_time_index: np.ndarray,
                              target_location_index: Optional[np.ndarray] = None,
                              alpha: float = 0.5) -> 'HybridGAMSSM':
    """
    Transfer LUR coefficients from source to target with fine-tuning.

    Args:
        source_model: Fitted source HybridGAMSSM
        target_X: Target spatial features
        target_y: Target observations
        target_time_index: Target time indices
        target_location_index: Target location indices
        alpha: Weight for source coefficients (0=target only, 1=source only)

    Returns:
        Target model with transferred coefficients
    """
    if not GAM_SSM_AVAILABLE:
        raise ImportError("gam_ssm_lur package required")

    # Create new model with same configuration
    target_model = HybridGAMSSM(
        n_splines=source_model.n_splines,
        gam_lam=source_model.gam_lam,
        state_dim=source_model.state_dim,
        em_max_iter=source_model.em_max_iter,
        em_tol=source_model.em_tol,
        scalability_mode=source_model.scalability_mode,
        regularization=source_model.regularization,
        confidence_level=source_model.confidence_level,
        random_state=source_model.random_state,
    )

    # Fit target model
    target_model.fit(target_X, target_y, target_time_index, target_location_index)

    # Transfer GAM coefficients with weighted averaging
    if alpha > 0 and hasattr(source_model, 'gam_') and hasattr(target_model, 'gam_'):
        try:
            source_coef = source_model.gam_.coef_
            target_coef = target_model.gam_.coef_

            # Weighted combination
            transferred_coef = alpha * source_coef + (1 - alpha) * target_coef
            target_model.gam_.coef_ = transferred_coef
        except AttributeError as e:
            print(f"Warning: Could not transfer LUR coefficients - API mismatch: {e}")
            print("Continuing without LUR coefficient transfer.")

    return target_model


def transfer_ssm_dynamics(source_model: 'HybridGAMSSM',
                         target_model: 'HybridGAMSSM',
                         beta: float = 0.5) -> 'HybridGAMSSM':
    """
    Transfer SSM state transition matrix and noise covariances.

    Args:
        source_model: Fitted source model
        target_model: Fitted target model
        beta: Weight for source parameters

    Returns:
        Target model with transferred SSM parameters
    """
    if not GAM_SSM_AVAILABLE:
        raise ImportError("gam_ssm_lur package required")

    # Transfer state transition matrix
    if hasattr(source_model, 'ssm_') and hasattr(target_model, 'ssm_'):
        try:
            # Try to access SSM parameters - API may vary
            F_source = source_model.ssm_.F
            F_target = target_model.ssm_.F

            # Weighted combination
            F_transferred = beta * F_source + (1 - beta) * F_target
            target_model.ssm_.F = F_transferred

            # Transfer process noise covariance
            Q_source = source_model.ssm_.Q
            Q_target = target_model.ssm_.Q
            Q_transferred = beta * Q_source + (1 - beta) * Q_target
            target_model.ssm_.Q = Q_transferred
        except AttributeError as e:
            # Handle API mismatch gracefully
            print(f"Warning: Could not transfer SSM dynamics - API mismatch: {e}")
            print("The gam_ssm_lur package may have a different SSM structure.")
            print("Continuing without SSM parameter transfer.")

    return target_model


def hybrid_transfer(source_model: 'HybridGAMSSM',
                   target_X: np.ndarray,
                   target_y: np.ndarray,
                   target_time_index: np.ndarray,
                   target_location_index: Optional[np.ndarray] = None,
                   spatial_weight: float = 0.5,
                   temporal_weight: float = 0.5,
                   fine_tune: bool = True,
                   fine_tune_iter: int = 20) -> Tuple['HybridGAMSSM', Dict]:
    """
    Complete hybrid transfer learning pipeline.

    Transfers both spatial (LUR) and temporal (SSM) components.

    Args:
        source_model: Fitted source model
        target_X: Target features
        target_y: Target observations
        target_time_index: Time indices
        target_location_index: Location indices
        spatial_weight: Weight for source LUR coefficients
        temporal_weight: Weight for source SSM parameters
        fine_tune: Whether to fine-tune after transfer
        fine_tune_iter: Fine-tuning iterations

    Returns:
        transferred_model: Model with transferred parameters
        transfer_info: Dictionary with transfer statistics
    """
    if not GAM_SSM_AVAILABLE:
        raise ImportError("gam_ssm_lur package required")

    # Step 1: Transfer LUR coefficients
    target_model = transfer_lur_coefficients(
        source_model, target_X, target_y,
        target_time_index, target_location_index,
        alpha=spatial_weight
    )

    # Step 2: Transfer SSM dynamics
    target_model = transfer_ssm_dynamics(
        source_model, target_model,
        beta=temporal_weight
    )

    # Step 3: Fine-tune (optional)
    if fine_tune:
        # Re-fit with transferred initialization
        original_em_iter = target_model.em_max_iter
        target_model.em_max_iter = fine_tune_iter
        target_model.fit(target_X, target_y, target_time_index, target_location_index)
        target_model.em_max_iter = original_em_iter

    # Collect transfer info
    transfer_info = {
        'spatial_weight': spatial_weight,
        'temporal_weight': temporal_weight,
        'fine_tuned': fine_tune,
        'source_n_locations': source_model.n_locations_ if hasattr(source_model, 'n_locations_') else None,
        'target_n_locations': target_model.n_locations_ if hasattr(target_model, 'n_locations_') else None,
    }

    return target_model, transfer_info


class TransferableGAMSSM:
    """
    Wrapper for HybridGAMSSM with transfer learning capabilities.

    Simplifies transfer learning workflow with save/load functionality.
    """

    def __init__(self, base_model: Optional['HybridGAMSSM'] = None, **model_kwargs):
        """
        Args:
            base_model: Existing model or None to create new
            **model_kwargs: Parameters for HybridGAMSSM
        """
        if not GAM_SSM_AVAILABLE:
            raise ImportError("gam_ssm_lur package required")

        if base_model is not None:
            self.model = base_model
        else:
            self.model = HybridGAMSSM(**model_kwargs)

        self.transfer_history = []

    def fit(self, X, y, time_index, location_index=None, **kwargs):
        """Fit the model."""
        self.model.fit(X, y, time_index, location_index, **kwargs)
        return self

    def transfer_from(self,
                     source: 'TransferableGAMSSM',
                     target_X, target_y, target_time_index,
                     target_location_index=None,
                     spatial_weight=0.5,
                     temporal_weight=0.5):
        """
        Transfer from source model.

        Args:
            source: Source TransferableGAMSSM
            target_X, target_y, target_time_index, target_location_index: Target data
            spatial_weight: LUR coefficient transfer weight
            temporal_weight: SSM dynamics transfer weight

        Returns:
            self with transferred parameters
        """
        self.model, transfer_info = hybrid_transfer(
            source.model, target_X, target_y,
            target_time_index, target_location_index,
            spatial_weight, temporal_weight
        )

        self.transfer_history.append({
            'source': source,
            'transfer_info': transfer_info
        })

        return self

    def predict(self, X, return_intervals=True):
        """Make predictions."""
        try:
            return self.model.predict(X, return_intervals=return_intervals)
        except TypeError:
            # API doesn't support return_intervals parameter
            return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """Evaluate predictions."""
        return self.model.evaluate(y_true, y_pred)


def get_transfer_summary(source_model: 'HybridGAMSSM',
                        target_model: 'HybridGAMSSM') -> Dict:
    """
    Summarize differences between source and target models.

    Returns:
        Dictionary with comparison metrics
    """
    summary = {}

    # GAM component comparison
    if hasattr(source_model, 'gam_') and hasattr(target_model, 'gam_'):
        coef_diff = np.linalg.norm(source_model.gam_.coef_ - target_model.gam_.coef_)
        summary['gam_coef_l2_diff'] = coef_diff

    # SSM component comparison
    if hasattr(source_model, 'ssm_') and hasattr(target_model, 'ssm_'):
        F_diff = np.linalg.norm(source_model.ssm_.F - target_model.ssm_.F, 'fro')
        summary['ssm_F_frobenius_diff'] = F_diff

    return summary
