"""
Real Model Transfer Learning Experiments
=========================================

Loads pre-trained Dublin models (FusionGP and GAM-SSM-LUR) and transfers them
to synthetic Cork data using OBTL and Prior Tempering paradigms.

This is the core experiment for the thesis chapter on transfer learning.
"""

import numpy as np
import torch
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gam_ssm_lur' / 'fusionGP2' / 'fusiongp' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gam_ssm_lur' / 'src'))

from src.transfer_methods.obtl import OBTLGaussianProcess
from src.transfer_methods.prior_tempering import transfer_with_tempering
from src.models.gp_model import predict_with_uncertainty
from src.evaluation.metrics import regression_metrics
import gpytorch


def generate_synthetic_cork_data(n_target=50, n_test=100, n_features=3, seed=42):
    """
    Generate synthetic Cork data for transfer learning experiments.

    Simulates Cork air quality measurements with domain shift from Dublin.

    Parameters
    ----------
    n_target : int
        Number of Cork training samples
    n_test : int
        Number of Cork test samples
    n_features : int
        Number of features (should match Dublin model: 3 for FusionGP = [x, y, time])
    seed : int
        Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Target domain (Cork): 3D spatiotemporal features [x, y, time]
    # x, y: spatial coordinates (normalized)
    # time: temporal coordinate (normalized)
    X_target = torch.randn(n_target, n_features) * 0.5 + 0.5  # Centered around Dublin

    # Add systematic offset for Cork (different location)
    X_target[:, :2] += 0.3  # Cork is offset from Dublin in space

    # Generate NO₂ concentrations with spatiotemporal pattern
    y_target = (
        15.0 +  # Base concentration
        5.0 * torch.sin(2 * np.pi * X_target[:, 0]) +  # Spatial pattern in x
        3.0 * torch.cos(2 * np.pi * X_target[:, 1]) +  # Spatial pattern in y
        2.0 * torch.sin(4 * np.pi * X_target[:, 2]) +  # Temporal pattern
        torch.randn(n_target) * 1.5  # Noise
    )

    # Test data (same distribution as Cork target)
    X_test = torch.randn(n_test, n_features) * 0.5 + 0.5
    X_test[:, :2] += 0.3  # Same Cork offset

    y_test = (
        15.0 +
        5.0 * torch.sin(2 * np.pi * X_test[:, 0]) +
        3.0 * torch.cos(2 * np.pi * X_test[:, 1]) +
        2.0 * torch.sin(4 * np.pi * X_test[:, 2]) +
        torch.randn(n_test) * 1.5
    )

    return {
        'target': {'X': X_target, 'y': y_target},
        'test': {'X': X_test, 'y': y_test}
    }


def load_dublin_fusiongp(model_path: str):
    """
    Load pre-trained Dublin FusionGP model.

    Parameters
    ----------
    model_path : str
        Path to saved FusionGP model (.pth file)

    Returns
    -------
    model : BaselineGP
        Loaded FusionGP model (simplified for transfer)
    likelihood : gpytorch.likelihoods.Likelihood
        Associated likelihood
    """
    print(f"\n📦 Loading Dublin FusionGP model from: {model_path}")

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


def load_dublin_gam_ssm_lur(gam_path: str, ssm_path: str, data_path: str):
    """
    Load pre-trained Dublin GAM-SSM-LUR model.

    Parameters
    ----------
    gam_path : str
        Path to GAM component (.pkl)
    ssm_path : str
        Path to SSM component (.pkl)
    data_path : str
        Path to training data (.npz)

    Returns
    -------
    model : dict
        Loaded GAM-SSM-LUR components
    data : dict
        Training data
    """
    print(f"\n📦 Loading Dublin GAM-SSM-LUR model")

    import pickle
    import numpy as np

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
    print(f"   ✓ Loaded training data: {data['X_train'].shape[0]} samples, {data['X_train'].shape[1]} features")

    model = {
        'gam': gam_model,
        'ssm': ssm_model,
        'type': 'GAM-SSM-LUR'
    }

    return model, data


def transfer_fusiongp_with_prior_tempering(
    source_model,
    source_likelihood,
    target_data: Dict,
    test_data: Dict,
    beta_values: list = [0.3, 0.5, 0.7, 1.0]
) -> Dict:
    """
    Transfer FusionGP from Dublin to Cork using Prior Tempering.

    Parameters
    ----------
    source_model : gpytorch.models.ExactGP
        Pre-trained Dublin FusionGP
    source_likelihood : gpytorch.likelihoods.Likelihood
        Source likelihood
    target_data : dict
        Cork training data
    test_data : dict
        Cork test data
    beta_values : list
        Temperature parameters to test

    Returns
    -------
    results : dict
        Transfer learning results for each beta
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT: FusionGP Transfer with Prior Tempering")
    print(f"{'='*70}")
    print(f"Source: Real Dublin FusionGP model")
    print(f"Target: Synthetic Cork data ({target_data['X'].shape[0]} samples)")
    print(f"Beta values: {beta_values}")

    results = []

    for beta in beta_values:
        print(f"\n  β = {beta:.2f}")

        # Transfer with Prior Tempering
        target_model, target_likelihood = transfer_with_tempering(
            source_gp=source_model,
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

        # Compute metrics
        metrics = regression_metrics(y_pred, test_data['y'].numpy())

        print(f"    RMSE: {metrics['rmse']:.2f} µg/m³")
        print(f"    MAE:  {metrics['mae']:.2f} µg/m³")
        print(f"    R²:   {metrics['r2']:.4f}")

        results.append({
            'beta': beta,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        })

    return {
        'experiment': 'FusionGP_Prior_Tempering',
        'source': 'Real Dublin FusionGP',
        'target': 'Synthetic Cork',
        'results': results,
        'best': min(results, key=lambda x: x['rmse'])
    }


def transfer_fusiongp_with_obtl(
    source_model,
    source_likelihood,
    source_data: Dict,
    target_data: Dict,
    test_data: Dict,
    delta_values: list = [0.3, 0.5, 0.7, 1.0]
) -> Dict:
    """
    Transfer FusionGP from Dublin to Cork using OBTL.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT: FusionGP Transfer with OBTL")
    print(f"{'='*70}")

    # Note: OBTL works on covariance structures
    # You'll need source data to extract covariance
    print("⚠️  OBTL requires source data - please provide Dublin training data")

    # Placeholder for OBTL transfer
    # This requires access to Dublin training data
    raise NotImplementedError(
        "OBTL transfer requires Dublin source data. "
        "Please provide X_source, y_source for covariance extraction."
    )


def main():
    """
    Run real model transfer learning experiments.
    """
    print("="*70)
    print("REAL MODEL TRANSFER LEARNING EXPERIMENTS")
    print("="*70)
    print("\nTransfer Scenario:")
    print("  Source: Pre-trained Dublin FusionGP (real data)")
    print("  Target: Synthetic Cork data")
    print("  Methods: Prior Tempering, OBTL")
    print("="*70)

    # Paths to saved models
    base_path = Path(__file__).parent.parent
    fusiongp_path = base_path / 'models' / 'fusiongp' / 'dublin' / 'fusiongp_model.pth'
    gam_path = base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'gam.pkl'
    ssm_path = base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'ssm.pkl'
    gam_data_path = base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'training_data.npz'

    # Generate synthetic Cork data
    print("\n📦 Generating synthetic Cork data...")
    cork_data = generate_synthetic_cork_data(n_target=50, n_test=100, seed=42)
    print(f"   Target: {cork_data['target']['X'].shape[0]} samples")
    print(f"   Test:   {cork_data['test']['X'].shape[0]} samples")

    # Load Dublin FusionGP
    try:
        dublin_fusiongp, dublin_likelihood = load_dublin_fusiongp(str(fusiongp_path))
    except Exception as e:
        print(f"\n❌ Error loading FusionGP: {e}")
        import traceback
        traceback.print_exc()
        return

    # Experiment 1: FusionGP with Prior Tempering
    results_pt = transfer_fusiongp_with_prior_tempering(
        dublin_fusiongp,
        dublin_likelihood,
        cork_data['target'],
        cork_data['test'],
        beta_values=[0.0, 0.3, 0.5, 0.7, 1.0]
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / 'results' / 'real_model_transfer'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': timestamp,
        'source': 'Real Dublin FusionGP',
        'target': 'Synthetic Cork',
        'fusiongp_prior_tempering': results_pt
    }

    json_file = output_dir / f'real_transfer_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✓ Results saved to: {json_file}")
    print(f"\n📊 Best Prior Tempering:")
    print(f"   β = {results_pt['best']['beta']}")
    print(f"   RMSE = {results_pt['best']['rmse']:.2f} µg/m³")
    print(f"   R² = {results_pt['best']['r2']:.4f}")


if __name__ == '__main__':
    main()
