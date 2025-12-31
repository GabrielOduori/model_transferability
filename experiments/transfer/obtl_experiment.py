"""
OBTL Transfer Experiment
=========================

Unified experiment class for OBTL transfer learning.
Works with both FusionGP and GAM-SSM-LUR models.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import linear_operator
from typing import Dict, Tuple, Any

# Add project root to path
base_path = Path(__file__).parent.parent.parent
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

from src.transfer_methods.obtl import OBTLGaussianProcess
from src.evaluation.metrics import regression_metrics
from src.models.gp_model import predict_with_uncertainty

from .base_transfer import BaseTransferExperiment


class OBTLExperiment(BaseTransferExperiment):
    """
    OBTL (Optimal Bayesian Transfer Learning) experiment.

    Supports both GP-based models (FusionGP) and GAM-SSM-LUR models.
    """

    def __init__(
        self,
        source_model=None,
        source_likelihood=None,
        source_data: Dict = None,
        model_type: str = 'fusiongp',
        n_inducing_points: int = 30,
        nu_0: float = 25.0
    ):
        """
        Initialize OBTL experiment.

        Args:
            source_model: Pre-trained source model (GP or GAM model dict)
            source_likelihood: Source likelihood (for GP models)
            source_data: Source training data (for GAM-SSM-LUR)
            model_type: 'fusiongp' or 'gam_ssm_lur'
            n_inducing_points: Number of inducing points for OBTL
            nu_0: Degrees of freedom for source prior
        """
        self.model_type = model_type.lower()
        self.n_inducing_points = n_inducing_points
        self.nu_0 = nu_0

        if self.model_type == 'fusiongp':
            experiment_name = "FusionGP_OBTL"
            source_name = "Real FusionGP (Source domain)"
            self.source_model = source_model
            self.source_likelihood = source_likelihood
            self.source_data = None

            # Extract source training data
            self.X_source = source_model.train_inputs[0]
            self.y_source = source_model.train_targets

        elif self.model_type == 'gam_ssm_lur':
            experiment_name = "GAM-SSM-LUR_OBTL"
            source_name = "Real GAM-SSM-LUR (Source domain)"
            self.gam_model = source_model['gam']
            self.ssm_model = source_model['ssm']
            self.source_data = source_data
            self.source_model = None
            self.source_likelihood = None

            # Use source data from GAM-SSM-LUR
            self.X_source = torch.tensor(source_data['X_train'], dtype=torch.float32)
            self.y_source = torch.tensor(source_data['y_train'], dtype=torch.float32)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        super().__init__(experiment_name, source_name)

        print(f"   Source data: {self.X_source.shape[0]} training points")

    def get_parameter_name(self) -> str:
        """Return parameter name for OBTL."""
        return 'delta'

    def run_single_transfer(
        self,
        param_value: float,
        target_data: Dict,
        test_data: Dict
    ) -> Dict:
        """
        Run single OBTL transfer.

        Args:
            param_value: Delta value
            target_data: Target domain training data
            test_data: Target domain test data

        Returns:
            Dict with metrics for this delta value
        """
        delta = param_value

        # Use relaxed settings for OBTL (covariance matrices can be ill-conditioned)
        with linear_operator.settings.max_cg_iterations(2000), \
             linear_operator.settings.cg_tolerance(0.1), \
             linear_operator.settings.max_cholesky_size(2000), \
             linear_operator.settings.cholesky_jitter(1e-3):  # Add more jitter for PSD

            # Initialize OBTL
            obtl = OBTLGaussianProcess(
                n_inducing_points=self.n_inducing_points,
                nu_0=self.nu_0,
                delta=delta
            )

            # Fit source domain
            obtl.fit_source(self.X_source, self.y_source, num_iter=100)

            # Transfer to target domain and get GP model
            target_model, target_likelihood = obtl.transfer_to_target(
                target_data['X'],
                target_data['y'],
                delta=delta,
                num_iter=200,
                return_gp=True
            )

            # Calculate transfer weights
            total_precision_weight = delta * obtl.nu_0 + target_data['X'].shape[0]
            weight_source = (delta * obtl.nu_0) / total_precision_weight
            weight_target = target_data['X'].shape[0] / total_precision_weight

            # Predict on test set
            y_pred, y_std = predict_with_uncertainty(
                target_model, target_likelihood, test_data['X']
            )

        # Compute metrics
        metrics = regression_metrics(y_pred, test_data['y'].numpy())

        return {
            'delta': delta,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'weight_source': float(weight_source),
            'weight_target': float(weight_target)
        }


def create_obtl_experiment(
    model_type: str,
    source_model=None,
    source_likelihood=None,
    gam_model=None,
    ssm_model=None,
    source_data: Dict = None,
    n_inducing_points: int = 30,
    nu_0: float = 25.0
) -> OBTLExperiment:
    """
    Factory function to create OBTL experiment.

    Args:
        model_type: 'fusiongp' or 'gam_ssm_lur'
        source_model: Pre-trained FusionGP model (for FusionGP)
        source_likelihood: Source likelihood (for FusionGP)
        gam_model: GAM component (for GAM-SSM-LUR)
        ssm_model: SSM component (for GAM-SSM-LUR)
        source_data: Source training data (for GAM-SSM-LUR)
        n_inducing_points: Number of inducing points
        nu_0: Degrees of freedom

    Returns:
        OBTLExperiment instance
    """
    if model_type.lower() == 'fusiongp':
        return OBTLExperiment(
            source_model=source_model,
            source_likelihood=source_likelihood,
            model_type='fusiongp',
            n_inducing_points=n_inducing_points,
            nu_0=nu_0
        )
    elif model_type.lower() == 'gam_ssm_lur':
        model_dict = {'gam': gam_model, 'ssm': ssm_model}
        return OBTLExperiment(
            source_model=model_dict,
            source_data=source_data,
            model_type='gam_ssm_lur',
            n_inducing_points=n_inducing_points,
            nu_0=nu_0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
