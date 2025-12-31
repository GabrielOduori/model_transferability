"""
Model Loading Utilities
========================

Load pre-trained source domain models for transfer learning.
"""

import torch
import pickle
import numpy as np
import gpytorch
from pathlib import Path
from typing import Tuple, Dict, Optional


class ModelLoader:
    """Load pre-trained source domain models."""

    @staticmethod
    def load_fusiongp(model_path: Path) -> Tuple:
        """
        Load pre-trained FusionGP model.

        Converts the full FusionGP model to a BaselineGP using inducing points
        as pseudo-training data for transfer learning compatibility.

        Args:
            model_path: Path to saved FusionGP model (.pth file)

        Returns:
            Tuple of (model, likelihood)
                - model: BaselineGP with loaded hyperparameters
                - likelihood: GaussianLikelihood

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"FusionGP model not found: {model_path}")

        print(f"\n🔄 Loading FusionGP model from: {model_path.name}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        print(f"   Model: FusionGP with {checkpoint['model_config']['n_inducing']} inducing points")
        print(f"   Kernel: {checkpoint['model_config']['kernel_type']}")

        # Extract inducing points as pseudo-training data for transfer
        inducing_points = checkpoint['model_state_dict']['variational_strategy.inducing_points']
        variational_mean = checkpoint['model_state_dict']['variational_strategy._variational_distribution.variational_mean']

        # Use a subset for BaselineGP (for transfer learning compatibility)
        n_pseudo = min(100, len(inducing_points))
        train_x = inducing_points[:n_pseudo, :]
        train_y = variational_mean[:n_pseudo]

        print(f"   Using {n_pseudo} inducing points as pseudo-training data")

        # Create BaselineGP model (compatible with our transfer methods)
        import sys
        from pathlib import Path as PathLib
        base_path = PathLib(__file__).parent.parent.parent
        if str(base_path) not in sys.path:
            sys.path.insert(0, str(base_path))

        from src.models.gp_model import BaselineGP

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(train_x, train_y, likelihood)

        # Load learned hyperparameters
        try:
            # Set lengthscales
            if 'covar_module.spatial_kernel.raw_lengthscale' in checkpoint['model_state_dict']:
                spatial_ls = checkpoint['model_state_dict']['covar_module.spatial_kernel.raw_lengthscale']
                model.covar_module.base_kernel.lengthscale = spatial_ls[:, :2]  # Spatial dims only

            # Set outputscale
            if 'covar_module.outputscale_param' in checkpoint['model_state_dict']:
                outputscale = checkpoint['model_state_dict']['covar_module.outputscale_param']
                model.covar_module.outputscale = outputscale

            # Set mean
            if 'mean_module.raw_constant' in checkpoint['model_state_dict']:
                mean_const = checkpoint['model_state_dict']['mean_module.raw_constant']
                model.mean_module.constant.data = mean_const

            print("   ✓ Loaded learned hyperparameters")
        except Exception as e:
            print(f"   ⚠️  Partial hyperparameter loading: {e}")

        model.eval()
        likelihood.eval()
        print("   ✓ FusionGP converted to BaselineGP for transfer")

        return model, likelihood

    @staticmethod
    def load_gam_ssm_lur(
        gam_path: Path,
        ssm_path: Path,
        data_path: Path
    ) -> Tuple[Dict, Dict]:
        """
        Load pre-trained GAM-SSM-LUR model.

        Args:
            gam_path: Path to GAM component (.pkl)
            ssm_path: Path to SSM component (.pkl)
            data_path: Path to training data (.npz)

        Returns:
            Tuple of (model, data)
                - model: Dict with 'gam', 'ssm', 'type' keys
                - data: Dict with training data

        Raises:
            FileNotFoundError: If any model file doesn't exist
        """
        gam_path = Path(gam_path)
        ssm_path = Path(ssm_path)
        data_path = Path(data_path)

        # Check files exist
        for path, name in [(gam_path, 'GAM'), (ssm_path, 'SSM'), (data_path, 'data')]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")

        print(f"\n🔄 Loading GAM-SSM-LUR model")

        # Load GAM component
        with open(gam_path, 'rb') as f:
            gam_model = pickle.load(f)
        print(f"   ✓ Loaded GAM component")

        # Load SSM component
        with open(ssm_path, 'rb') as f:
            ssm_model = pickle.load(f)
        print(f"   ✓ Loaded SSM component")

        # Load training data
        data_npz = np.load(data_path)
        data = {
            'X_train': data_npz['X_train'],
            'y_train': data_npz['y_train'],
            'y_matrix': data_npz['y_matrix'],
            'residual_matrix': data_npz['residual_matrix']
        }
        print(f"   ✓ Loaded training data: {data['X_train'].shape[0]} samples, "
              f"{data['X_train'].shape[1]} features")

        model = {
            'gam': gam_model,
            'ssm': ssm_model,
            'type': 'GAM-SSM-LUR'
        }

        return model, data

    @staticmethod
    def verify_model_files(config) -> Dict[str, bool]:
        """
        Verify that all required model files exist.

        Args:
            config: ExperimentConfig instance

        Returns:
            Dict mapping model names to existence status
        """
        status = {
            'fusiongp': config.fusiongp_path.exists(),
            'gam': config.gam_path.exists(),
            'ssm': config.ssm_path.exists(),
            'gam_data': config.gam_data_path.exists(),
        }

        print("\n📋 Model Files Status:")
        for name, exists in status.items():
            symbol = "✓" if exists else "✗"
            print(f"   {symbol} {name}: {exists}")

        return status
