"""
Prior Tempering Transfer Experiment
====================================

Unified experiment class for Prior Tempering transfer learning.
Works with both FusionGP and GAM-SSM-LUR models.
"""

import sys
from pathlib import Path
import torch
import gpytorch
import numpy as np
from typing import Dict, Tuple, Any

# Add project root to path
base_path = Path(__file__).parent.parent.parent
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

from src.transfer_methods.prior_tempering import transfer_with_tempering
from src.evaluation.metrics import regression_metrics
from src.models.gp_model import BaselineGP, predict_with_uncertainty

from .base_transfer import BaseTransferExperiment


class PriorTemperingExperiment(BaseTransferExperiment):
    """
    Prior Tempering transfer learning experiment.

    Supports both GP-based models (FusionGP) and GAM-SSM-LUR models.
    """

    def __init__(
        self,
        source_model,
        source_likelihood=None,
        source_data: Dict = None,
        model_type: str = 'fusiongp'
    ):
        """
        Initialize Prior Tempering experiment.

        Args:
            source_model: Pre-trained source model (GP or GAM)
            source_likelihood: Source likelihood (for GP models)
            source_data: Source training data (for GAM-SSM-LUR)
            model_type: 'fusiongp' or 'gam_ssm_lur'
        """
        self.model_type = model_type.lower()

        if self.model_type == 'fusiongp':
            experiment_name = "FusionGP_Prior_Tempering"
            source_name = "Real FusionGP (Source domain)"
            self.source_model = source_model
            self.source_likelihood = source_likelihood
            self.source_data = None

        elif self.model_type == 'gam_ssm_lur':
            experiment_name = "GAM-SSM-LUR_Prior_Tempering"
            source_name = "Real GAM-SSM-LUR (Source domain)"
            self.gam_model = source_model['gam']
            self.ssm_model = source_model['ssm']
            self.source_data = source_data
            self.source_model = None
            self.source_likelihood = None

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        super().__init__(experiment_name, source_name)

    def get_parameter_name(self) -> str:
        """Return parameter name for Prior Tempering."""
        return 'lambda'  # Using lambda for consistency (β is temperature parameter)

    def run_single_transfer(
        self,
        param_value: float,
        target_data: Dict,
        test_data: Dict
    ) -> Dict:
        """
        Run single Prior Tempering transfer.

        Args:
            param_value: Lambda (β) value
            target_data: Target domain training data
            test_data: Target domain test data

        Returns:
            Dict with metrics for this lambda value
        """
        beta = param_value

        if self.model_type == 'fusiongp':
            # Transfer FusionGP with Prior Tempering
            target_model, target_likelihood = transfer_with_tempering(
                source_gp=self.source_model,
                target_x=target_data['X'],
                target_y=target_data['y'],
                beta=beta,
                num_iter=200,
                verbose=False
            )

            # Predict on test set
            y_pred, y_std = predict_with_uncertainty(
                target_model, target_likelihood, test_data['X']
            )

        elif self.model_type == 'gam_ssm_lur':
            # Transfer GAM-SSM-LUR with Prior Tempering
            # Extract Source model characteristics
            source_residuals = self.source_data['residual_matrix'].flatten()
            source_noise_var = float(np.var(source_residuals[~np.isnan(source_residuals)]))

            X_source = self.source_data['X_train']
            source_spatial_scale = float(np.std(X_source[:, :2]))

            # Create source hyperparameters dict for tempering
            source_hyperparams = {
                'lengthscale': torch.tensor([[source_spatial_scale] * target_data['X'].shape[1]]),
                'outputscale': torch.tensor(1.0),
                'noise': torch.tensor(source_noise_var),
                'mean_constant': torch.tensor(float(np.nanmean(self.source_data['y_train'])))
            }

            # Create baseline GP for target
            target_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            target_model = BaselineGP(target_data['X'], target_data['y'], target_likelihood)

            # Apply Source knowledge with tempering
            if beta > 0:
                lengthscale_scale = torch.exp(torch.tensor(beta - 1.0))
                lengthscale_value = source_hyperparams['lengthscale'] * lengthscale_scale
                lengthscale_clamped = torch.clamp(lengthscale_value, min=0.01, max=10.0)

                # Use initialize() method instead of direct assignment to avoid constraint violations
                target_model.covar_module.base_kernel.initialize(lengthscale=lengthscale_clamped)

                noise_scale = 0.5 + 0.5 * beta
                noise_value = source_hyperparams['noise'] * noise_scale
                noise_clamped = torch.clamp(noise_value, min=0.01, max=10.0)

                # Use initialize() method for noise
                target_likelihood.initialize(noise=noise_clamped)

                # Initialize mean constant properly
                target_model.mean_module.initialize(constant=source_hyperparams['mean_constant'])

            # Train with tempered prior
            target_model.train()
            target_likelihood.train()

            optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(target_likelihood, target_model)

            for i in range(200):
                optimizer.zero_grad()
                output = target_model(target_data['X'])
                loss = -mll(output, target_data['y'])

                if torch.isnan(loss) or torch.isinf(loss):
                    break

                loss.backward()
                optimizer.step()

            target_model.eval()
            target_likelihood.eval()

            # Predict on test set
            y_pred, y_std = predict_with_uncertainty(
                target_model, target_likelihood, test_data['X']
            )

        # Compute metrics
        metrics = regression_metrics(y_pred, test_data['y'].numpy())

        return {
            'lambda': beta,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        }


def create_prior_tempering_experiment(
    model_type: str,
    source_model=None,
    source_likelihood=None,
    gam_model=None,
    ssm_model=None,
    source_data: Dict = None
) -> PriorTemperingExperiment:
    """
    Factory function to create Prior Tempering experiment.

    Args:
        model_type: 'fusiongp' or 'gam_ssm_lur'
        source_model: Pre-trained FusionGP model (for FusionGP)
        source_likelihood: Source likelihood (for FusionGP)
        gam_model: GAM component (for GAM-SSM-LUR)
        ssm_model: SSM component (for GAM-SSM-LUR)
        source_data: Source training data (for GAM-SSM-LUR)

    Returns:
        PriorTemperingExperiment instance
    """
    if model_type.lower() == 'fusiongp':
        return PriorTemperingExperiment(
            source_model=source_model,
            source_likelihood=source_likelihood,
            model_type='fusiongp'
        )
    elif model_type.lower() == 'gam_ssm_lur':
        model_dict = {'gam': gam_model, 'ssm': ssm_model}
        return PriorTemperingExperiment(
            source_model=model_dict,
            source_data=source_data,
            model_type='gam_ssm_lur'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
