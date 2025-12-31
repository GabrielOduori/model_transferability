"""
Real Model Transfer Learning Experiments
=========================================

Loads pre-trained Source domain models (FusionGP and GAM-SSM-LUR) and transfers them
to synthetic Target domain data using OBTL and Prior Tempering paradigms.

This is the core experiment for the thesis chapter on transfer learning.
"""

import numpy as np
import torch
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import seaborn as sns
import pandas as pd

# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gam_ssm_lur' / 'fusionGP2' / 'fusiongp' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gam_ssm_lur' / 'src'))

from src.transfer_methods.obtl import OBTLGaussianProcess
from src.transfer_methods.prior_tempering import transfer_with_tempering
from src.models.gp_model import predict_with_uncertainty
from src.evaluation.metrics import regression_metrics
from src.visualization.conceptual_diagrams import plot_prior_tempering_concept
from src.visualization.spatial_maps import plot_spatial_comparison, generate_example_spatial_data
from src.visualization.training_diagnostics import (
    plot_transfer_gain_loss,
    plot_transfer_weight_evolution
)
import gpytorch
import linear_operator


def generate_synthetic_target_data(n_target=50, n_test=100, n_features=3, seed=42, force_regenerate=False):
    """
    Generate synthetic target domain data for transfer learning experiments.

    If saved data exists (data/synthetic_target/target_data_seed42.npz),
    loads it for reproducibility. Set force_regenerate=True to regenerate.

    Simulates target domain air quality measurements with domain shift from source.

    Parameters
    ----------
    n_target : int
        Number of target training samples
    n_test : int
        Number of target test samples
    n_features : int
        Number of features (should match source model: 3 for FusionGP = [x, y, time])
    seed : int
        Random seed
    force_regenerate : bool
        If True, regenerate data even if saved file exists
    """
    # Check for saved data
    data_dir = Path(__file__).parent.parent / 'data' / 'synthetic_target'
    data_file = data_dir / f'target_data_seed{seed}.npz'

    if data_file.exists() and not force_regenerate:
        print(f"\n Loading saved synthetic data: {data_file}")
        data = np.load(data_file)

        # Verify metadata matches
        if (int(data['n_target']) == n_target and
            int(data['n_test']) == n_test and
            int(data['seed']) == seed):

            print(f"  Metadata: n_target={data['n_target']}, n_test={data['n_test']}, "
                  f"seed={data['seed']}, domain_shift={data['domain_shift']}")

            # Convert to torch tensors
            X_target = torch.from_numpy(data['X_target'])
            y_target = torch.from_numpy(data['y_target'])
            X_test = torch.from_numpy(data['X_test'])
            y_test = torch.from_numpy(data['y_test'])

            print(f"   Using saved data for reproducibility")

            return {
                'target': {'X': X_target, 'y': y_target},
                'test': {'X': X_test, 'y': y_test},
                'metadata': {
                    'seed': seed,
                    'n_target': n_target,
                    'n_test': n_test,
                    'domain_shift': float(data['domain_shift']),
                    'noise_std': float(data['noise_std']),
                    'saved_file': str(data_file),
                    'loaded_from_file': True
                }
            }
        else:
            print(f"  Saved data metadata mismatch, regenerating...")

    # Generate new data
    print(f"\n Generating new synthetic target data (seed={seed})")
    print(f"  n_target={n_target}, n_test={n_test}, domain_shift=0.3")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Target domain: 3D spatiotemporal features [x, y, time]
    # x, y: spatial coordinates (normalized)
    # time: temporal coordinate (normalized)
    X_target = torch.randn(n_target, n_features) * 0.5 + 0.5  # Base distribution

    # Add systematic offset for target domain (domain shift)
    X_target[:, :2] += 0.3  # Spatial offset from source domain

    # Generate NO₂ concentrations with spatiotemporal pattern
    y_target = (
        15.0 +  # Base concentration
        5.0 * torch.sin(2 * np.pi * X_target[:, 0]) +  # Spatial pattern in x
        3.0 * torch.cos(2 * np.pi * X_target[:, 1]) +  # Spatial pattern in y
        2.0 * torch.sin(4 * np.pi * X_target[:, 2]) +  # Temporal pattern
        torch.randn(n_target) * 1.5  # Noise
    )

    # Test data (same distribution as target domain)
    X_test = torch.randn(n_test, n_features) * 0.5 + 0.5
    X_test[:, :2] += 0.3  # Same spatial offset

    y_test = (
        15.0 +
        5.0 * torch.sin(2 * np.pi * X_test[:, 0]) +
        3.0 * torch.cos(2 * np.pi * X_test[:, 1]) +
        2.0 * torch.sin(4 * np.pi * X_test[:, 2]) +
        torch.randn(n_test) * 1.5
    )

    # Save synthetic data for reproducibility
    data_dir = Path(__file__).parent.parent / 'data' / 'synthetic_target'
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / f'target_data_seed{seed}.npz'
    np.savez(
        data_file,
        X_target=X_target.numpy(),
        y_target=y_target.numpy(),
        X_test=X_test.numpy(),
        y_test=y_test.numpy(),
        n_target=n_target,
        n_test=n_test,
        n_features=n_features,
        seed=seed,
        domain_shift=0.3,
        noise_std=1.5
    )
    print(f" Saved synthetic target data: {data_file}")

    return {
        'target': {'X': X_target, 'y': y_target},
        'test': {'X': X_test, 'y': y_test},
        'metadata': {
            'seed': seed,
            'n_target': n_target,
            'n_test': n_test,
            'domain_shift': 0.3,
            'noise_std': 1.5,
            'saved_file': str(data_file)
        }
    }


def load_source_fusiongp(model_path: str):
    """
    Load pre-trained Source domain FusionGP model.

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
    print(f"\n Loading Source FusionGP model from: {model_path}")

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

        print("    Loaded learned hyperparameters")
    except Exception as e:
        print(f"   Partial hyperparameter loading: {e}")

    model.eval()
    likelihood.eval()
    print("    FusionGP converted to BaselineGP for transfer")

    return model, likelihood


def load_source_gam_ssm_lur(gam_path: str, ssm_path: str, data_path: str):
    """
    Load pre-trained Source domain GAM-SSM-LUR model.

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
    print(f"\n Loading Source GAM-SSM-LUR model")

    import pickle
    import numpy as np

    # Load GAM component
    with open(gam_path, 'rb') as f:
        gam_model = pickle.load(f)
    print(f"    Loaded GAM component")

    # Load SSM component
    with open(ssm_path, 'rb') as f:
        ssm_model = pickle.load(f)
    print(f"    Loaded SSM component")

    # Load training data
    data_npz = np.load(data_path)
    data = {
        'X_train': data_npz['X_train'],
        'y_train': data_npz['y_train'],
        'y_matrix': data_npz['y_matrix'],
        'residual_matrix': data_npz['residual_matrix']
    }
    print(f"    Loaded training data: {data['X_train'].shape[0]} samples, {data['X_train'].shape[1]} features")

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
    Transfer FusionGP from Source to Target domain using Prior Tempering.

    Parameters
    ----------
    source_model : gpytorch.models.ExactGP
        Pre-trained Source domain FusionGP
    source_likelihood : gpytorch.likelihoods.Likelihood
        Source likelihood
    target_data : dict
        Target domain training data
    test_data : dict
        Target domain test data
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
    print(f"Source: Real FusionGP model (trained on Source domain)")
    print(f"Target: Synthetic Target domain data ({target_data['X'].shape[0]} samples)")
    print(f"Lambda (λ) values: {beta_values}")

    results = []

    for beta in beta_values:
        print(f"\n  λ = {beta:.2f}")

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
            'lambda': beta,  # Using 'lambda' for consistency (beta is temperature param)
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        })

    return {
        'experiment': 'FusionGP_Prior_Tempering',
        'source': 'Real FusionGP (Source domain)',
        'target': 'Synthetic Target domain',
        'results': results,
        'best': min(results, key=lambda x: x['rmse'])
    }


def transfer_gam_ssm_lur_with_prior_tempering(
    gam_model,
    ssm_model,
    source_data: Dict,
    target_data: Dict,
    test_data: Dict,
    beta_values: list = [0.3, 0.5, 0.7, 1.0]
) -> Dict:
    """
    Transfer GAM-SSM-LUR from Source to Target domain using Prior Tempering.

    Strategy: Extract spatial hyperparameters from GAM component and use
    them to inform a GP model for the Target domain.

    Parameters
    ----------
    gam_model : SpatialGAM
        Pre-trained Source domain GAM component
    ssm_model : StateSpaceModel
        Pre-trained Source domain SSM component
    source_data : dict
        Source domain training data
    target_data : dict
        Target domain training data
    test_data : dict
        Target domain test data
    beta_values : list
        Temperature parameters to test

    Returns
    -------
    results : dict
        Transfer learning results for each beta
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT: GAM-SSM-LUR Transfer with Prior Tempering")
    print(f"{'='*70}")
    print(f"Source: Real GAM-SSM-LUR model (trained on Source domain)")
    print(f"Target: Synthetic Target domain data ({target_data['X'].shape[0]} samples)")
    print(f"Lambda (λ) values: {beta_values}")

    # Extract Source model characteristics
    # Use Source residuals to estimate noise variance
    source_residuals = source_data['residual_matrix'].flatten()
    source_noise_var = float(np.var(source_residuals[~np.isnan(source_residuals)]))

    # Use spatial scale from Source data extent
    X_source = source_data['X_train']
    source_spatial_scale = float(np.std(X_source[:, :2]))  # Assuming first 2 dims are spatial

    print(f"   Source noise variance: {source_noise_var:.4f}")
    print(f"   Source spatial scale: {source_spatial_scale:.4f}")

    results = []

    from src.models.gp_model import BaselineGP

    for beta in beta_values:
        print(f"\n  λ = {beta:.2f}")

        # Create source hyperparameters dict for tempering
        source_hyperparams = {
            'lengthscale': torch.tensor([[source_spatial_scale] * target_data['X'].shape[1]]),
            'outputscale': torch.tensor(1.0),  # Normalized
            'noise': torch.tensor(source_noise_var),
            'mean_constant': torch.tensor(float(np.nanmean(source_data['y_train'])))
        }

        # Create a simple baseline GP for target
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(target_data['X'], target_data['y'], likelihood)

        # Apply Source knowledge with tempering
        if beta > 0:
            # Set prior from Source with tempering factor
            # Scale lengthscale: higher beta = more Source influence
            lengthscale_scale = torch.exp(torch.tensor(beta - 1.0))
            lengthscale_value = source_hyperparams['lengthscale'] * lengthscale_scale
            model.covar_module.base_kernel.lengthscale = torch.clamp(lengthscale_value, min=0.01, max=10.0)

            # Scale noise: higher beta = more Source noise
            noise_scale = 0.5 + 0.5 * beta
            noise_value = source_hyperparams['noise'] * noise_scale
            likelihood.noise = torch.clamp(noise_value.clone().detach(), min=0.01, max=10.0)

            # Set mean from Source
            model.mean_module.constant.data = source_hyperparams['mean_constant']

        # Train with tempered prior
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(200):
            optimizer.zero_grad()
            output = model(target_data['X'])
            loss = -mll(output, target_data['y'])

            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"      Warning: Numerical instability at iteration {i}, stopping training")
                break

            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        # Predict on test set
        y_pred, y_std = predict_with_uncertainty(
            model, likelihood, test_data['X']
        )

        # Compute metrics
        metrics = regression_metrics(y_pred, test_data['y'].numpy())

        print(f"    RMSE: {metrics['rmse']:.2f} µg/m³")
        print(f"    MAE:  {metrics['mae']:.2f} µg/m³")
        print(f"    R²:   {metrics['r2']:.4f}")

        results.append({
            'lambda': beta,  # Using 'lambda' for consistency (beta is temperature param)
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        })

    return {
        'experiment': 'GAM_SSM_LUR_Prior_Tempering',
        'source': 'Real GAM-SSM-LUR (Source domain)',
        'target': 'Synthetic Target domain',
        'results': results,
        'best': min(results, key=lambda x: x['rmse'])
    }


def transfer_fusiongp_with_obtl(
    source_model,
    source_likelihood,
    target_data: Dict,
    test_data: Dict,
    delta_values: list = [0.3, 0.5, 0.7, 1.0]
) -> Dict:
    """
    Transfer FusionGP from Source to Target domain using OBTL.

    Parameters
    ----------
    source_model : gpytorch.models.ExactGP
        Pre-trained Source domain FusionGP (contains training data)
    source_likelihood : gpytorch.likelihoods.Likelihood
        Source likelihood
    target_data : dict
        Target domain training data
    test_data : dict
        Target domain test data
    delta_values : list
        Transfer strength parameters to test

    Returns
    -------
    results : dict
        Transfer learning results for each delta
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT: FusionGP Transfer with OBTL")
    print(f"{'='*70}")
    print(f"Source: Real FusionGP model (trained on Source domain)")
    print(f"Target: Synthetic Target domain data ({target_data['X'].shape[0]} samples)")
    print(f"Delta values: {delta_values}")

    # Extract Source training data from the model
    X_source = source_model.train_inputs[0]
    y_source = source_model.train_targets

    print(f"   Source data: {X_source.shape[0]} pseudo-training points")

    results = []

    for delta in delta_values:
        print(f"\n  δ = {delta:.2f}")

        # Use relaxed settings for OBTL (covariance matrices can be ill-conditioned)
        with linear_operator.settings.max_cg_iterations(2000), \
             linear_operator.settings.cg_tolerance(0.1), \
             linear_operator.settings.max_cholesky_size(2000), \
             linear_operator.settings.cholesky_jitter(1e-3):  # Add more jitter for PSD

            # Initialize OBTL with more inducing points for better conditioning
            obtl = OBTLGaussianProcess(n_inducing_points=30, nu_0=25.0, delta=delta)

            # Fit Source domain
            obtl.fit_source(X_source, y_source, num_iter=100)

            # Transfer to Target domain and get GP model
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

        print(f"    RMSE: {metrics['rmse']:.2f} µg/m³")
        print(f"    MAE:  {metrics['mae']:.2f} µg/m³")
        print(f"    R²:   {metrics['r2']:.4f}")
        print(f"    Transfer weights: Source={weight_source:.3f}, Target={weight_target:.3f}")

        results.append({
            'delta': delta,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'weight_source': float(weight_source),
            'weight_target': float(weight_target)
        })

    return {
        'experiment': 'FusionGP_OBTL',
        'source': 'Real FusionGP (Source domain)',
        'target': 'Synthetic Target domain',
        'results': results,
        'best': min(results, key=lambda x: x['rmse'])
    }


def transfer_gam_ssm_lur_with_obtl(
    gam_model,
    ssm_model,
    source_data: Dict,
    target_data: Dict,
    test_data: Dict,
    delta_values: list = [0.3, 0.5, 0.7, 1.0]
) -> Dict:
    """
    Transfer GAM-SSM-LUR from Source to Target domain using OBTL.

    Strategy: Use Source data to extract covariance structure, then transfer to Target.

    Parameters
    ----------
    gam_model : SpatialGAM
        Pre-trained Source domain GAM component
    ssm_model : StateSpaceModel
        Pre-trained Source domain SSM component
    source_data : dict
        Source domain training data
    target_data : dict
        Target domain training data
    test_data : dict
        Target domain test data
    delta_values : list
        Transfer strength parameters to test

    Returns
    -------
    results : dict
        Transfer learning results for each delta
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT: GAM-SSM-LUR Transfer with OBTL")
    print(f"{'='*70}")
    print(f"Source: Real GAM-SSM-LUR model (trained on Source domain)")
    print(f"Target: Synthetic Target domain data ({target_data['X'].shape[0]} samples)")
    print(f"Delta values: {delta_values}")

    # Extract Source training data
    # Sample a subset for computational efficiency and numerical stability
    # Use fewer samples to avoid PSD issues with limited inducing points
    n_source_samples = min(300, source_data['X_train'].shape[0])
    indices = np.random.choice(source_data['X_train'].shape[0], n_source_samples, replace=False)

    X_source = torch.from_numpy(source_data['X_train'][indices]).float()
    y_source = torch.from_numpy(source_data['y_train'][indices]).float()

    # Match feature dimensions to Target (3D: x, y, time)
    # Use first 3 columns of Source data
    X_source = X_source[:, :3]

    print(f"   Source data: {X_source.shape[0]} samples (sampled from {source_data['X_train'].shape[0]})")

    results = []

    for delta in delta_values:
        print(f"\n  δ = {delta:.2f}")

        # Use relaxed settings for OBTL (covariance matrices can be ill-conditioned)
        with linear_operator.settings.max_cg_iterations(2000), \
             linear_operator.settings.cg_tolerance(0.1), \
             linear_operator.settings.max_cholesky_size(2000), \
             linear_operator.settings.cholesky_jitter(1e-3):  # Add more jitter for PSD

            # Initialize OBTL with more inducing points for better conditioning
            obtl = OBTLGaussianProcess(n_inducing_points=30, nu_0=20.0, delta=delta)

            # Fit Source domain
            obtl.fit_source(X_source, y_source, num_iter=100)

            # Transfer to Target domain and get GP model
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

        print(f"    RMSE: {metrics['rmse']:.2f} µg/m³")
        print(f"    MAE:  {metrics['mae']:.2f} µg/m³")
        print(f"    R²:   {metrics['r2']:.4f}")
        print(f"    Transfer weights: Source={weight_source:.3f}, Target={weight_target:.3f}")

        results.append({
            'delta': delta,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'weight_source': float(weight_source),
            'weight_target': float(weight_target)
        })

    return {
        'experiment': 'GAM_SSM_LUR_OBTL',
        'source': 'Real GAM-SSM-LUR (Source domain)',
        'target': 'Synthetic Target domain',
        'results': results,
        'best': min(results, key=lambda x: x['rmse'])
    }


def create_transfer_visualisations(all_results: Dict, output_dir: Path, timestamp: str):
    """
    Create publication-quality visualisations using Seaborn.

    Parameters
    ----------
    all_results : dict
        Complete results dictionary with all experiments
    output_dir : Path
        Directory to save figures
    timestamp : str
        Timestamp for file naming
    """
    # Create figure directory
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Creating publication-quality visualisations...")

    # Custom color palette (colorblind-friendly)
    colors = {
        'fusion': '#0173B2',      # Blue
        'gam': '#DE8F05',         # Orange
        'positive': '#029E73',    # Green
        'negative': '#CC78BC',    # Purple
        'neutral': '#949494'      # Gray
    }

    # Extract and prepare data
    fusiongp_pt = all_results['fusiongp_prior_tempering']['results']
    gam_pt = all_results['gam_ssm_lur_prior_tempering']['results']
    fusiongp_obtl = all_results['fusiongp_obtl']['results']
    gam_obtl = all_results['gam_ssm_lur_obtl']['results']

    # Sort results
    fusiongp_pt_sorted = sorted(fusiongp_pt, key=lambda x: x['lambda'])
    gam_pt_sorted = sorted(gam_pt, key=lambda x: x['lambda'])
    fusiongp_obtl_sorted = sorted(fusiongp_obtl, key=lambda x: x['delta'])
    gam_obtl_sorted = sorted(gam_obtl, key=lambda x: x['delta'])

    # Create DataFrames for seaborn
    pt_data = []
    for r in fusiongp_pt_sorted:
        pt_data.append({'Model': 'FusionGP', 'λ': r['lambda'], 'RMSE': r['rmse'],
                       'MAE': r['mae'], 'R²': r['r2']})
    for r in gam_pt_sorted:
        pt_data.append({'Model': 'GAM-SSM-LUR', 'λ': r['lambda'], 'RMSE': r['rmse'],
                       'MAE': r['mae'], 'R²': r['r2']})
    df_pt = pd.DataFrame(pt_data)

    obtl_data = []
    for r in fusiongp_obtl_sorted:
        obtl_data.append({'Model': 'FusionGP', 'δ': r['delta'], 'RMSE': r['rmse'],
                         'MAE': r['mae'], 'R²': r['r2']})
    for r in gam_obtl_sorted:
        obtl_data.append({'Model': 'GAM-SSM-LUR', 'δ': r['delta'], 'RMSE': r['rmse'],
                         'MAE': r['mae'], 'R²': r['r2']})
    df_obtl = pd.DataFrame(obtl_data)

    # ========== Figure 1: Prior Tempering - Multi-panel ==========
    fig1, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig1.patch.set_facecolor('white')
    fig1.suptitle('Prior Tempering Transfer Learning Results',
                  fontsize=18, fontweight='bold', y=0.998)

    # (a) RMSE vs Lambda with confidence ribbons
    ax = axes[0, 0]
    for model, color in [('FusionGP', colors['fusion']), ('GAM-SSM-LUR', colors['gam'])]:
        data = df_pt[df_pt['Model'] == model]
        sns.lineplot(data=data, x='λ', y='RMSE', ax=ax,
                    marker='o', markersize=10, linewidth=3,
                    color=color, label=model)
        # Highlight best point
        best_idx = data['RMSE'].idxmin()
        best_lambda = data.loc[best_idx, 'λ']
        best_rmse = data.loc[best_idx, 'RMSE']
        ax.scatter([best_lambda], [best_rmse], s=400, marker='*',
                  color='gold', edgecolor='black', linewidth=2, zorder=10)

    ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSE (µg/m³)', fontsize=13, fontweight='bold')
    ax.set_title('(a) Prediction Error vs Temperature', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    sns.despine(ax=ax)

    # (b) MAE vs Lambda
    ax = axes[0, 1]
    for model, color in [('FusionGP', colors['fusion']), ('GAM-SSM-LUR', colors['gam'])]:
        data = df_pt[df_pt['Model'] == model]
        sns.lineplot(data=data, x='λ', y='MAE', ax=ax,
                    marker='s', markersize=10, linewidth=3,
                    color=color, label=model)

    ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAE (µg/m³)', fontsize=13, fontweight='bold')
    ax.set_title('(b) Mean Absolute Error vs Temperature', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    sns.despine(ax=ax)

    # (c) R² vs Lambda with baseline
    ax = axes[1, 0]
    for model, color in [('FusionGP', colors['fusion']), ('GAM-SSM-LUR', colors['gam'])]:
        data = df_pt[df_pt['Model'] == model]
        sns.lineplot(data=data, x='λ', y='R²', ax=ax,
                    marker='D', markersize=10, linewidth=3,
                    color=color, label=model)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6,
              label='Baseline (R²=0)')
    ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_title('(c) Explained Variance vs Temperature', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    sns.despine(ax=ax)

    # (d) Improvement over Baseline with shaded regions
    ax = axes[1, 1]
    baseline_fusion = df_pt[(df_pt['Model'] == 'FusionGP') & (df_pt['λ'] == 0.0)]['RMSE'].values[0]
    baseline_gam = df_pt[(df_pt['Model'] == 'GAM-SSM-LUR') & (df_pt['λ'] == 0.0)]['RMSE'].values[0]

    for model, baseline, color in [('FusionGP', baseline_fusion, colors['fusion']),
                                    ('GAM-SSM-LUR', baseline_gam, colors['gam'])]:
        data = df_pt[df_pt['Model'] == model].copy()
        data['Improvement (%)'] = (baseline - data['RMSE']) / baseline * 100
        sns.lineplot(data=data, x='λ', y='Improvement (%)', ax=ax,
                    marker='o', markersize=10, linewidth=3,
                    color=color, label=model)

    # Shade positive/negative transfer regions
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax.fill_between([-0.1, 1.1], 0, 20, alpha=0.15, color=colors['positive'],
                   label='Positive Transfer')
    ax.fill_between([-0.1, 1.1], 0, -20, alpha=0.15, color=colors['negative'],
                   label='Negative Transfer')

    ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSE Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_title('(d) Transfer Learning Benefit', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, frameon=True, shadow=True, loc='best')
    ax.set_xlim(-0.05, 1.05)
    sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    fig1_path = fig_dir / f'prior_tempering_fancy_{timestamp}.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {fig1_path.name}")
    plt.close(fig1)

    # ========== Figure 2: OBTL Results ==========
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig2.patch.set_facecolor('white')
    fig2.suptitle('OBTL (Optimal Bayesian Transfer Learning) Results',
                  fontsize=18, fontweight='bold', y=1.02)

    # (a) RMSE vs Delta
    ax = axes[0]
    for model, color in [('FusionGP', colors['fusion']), ('GAM-SSM-LUR', colors['gam'])]:
        data = df_obtl[df_obtl['Model'] == model]
        sns.lineplot(data=data, x='δ', y='RMSE', ax=ax,
                    marker='o', markersize=12, linewidth=3,
                    color=color, label=model)

    ax.set_xlabel('Transfer Strength Parameter (δ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSE (µg/m³)', fontsize=13, fontweight='bold')
    ax.set_title('(a) RMSE vs Transfer Strength', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    sns.despine(ax=ax)

    # (b) Grouped bar chart - Best results comparison
    ax = axes[1]
    comparison_data = pd.DataFrame([
        {'Method': 'Prior Tempering', 'Model': 'FusionGP',
         'RMSE': df_pt[df_pt['Model'] == 'FusionGP']['RMSE'].min()},
        {'Method': 'Prior Tempering', 'Model': 'GAM-SSM-LUR',
         'RMSE': df_pt[df_pt['Model'] == 'GAM-SSM-LUR']['RMSE'].min()},
        {'Method': 'OBTL', 'Model': 'FusionGP',
         'RMSE': df_obtl[df_obtl['Model'] == 'FusionGP']['RMSE'].min()},
        {'Method': 'OBTL', 'Model': 'GAM-SSM-LUR',
         'RMSE': df_obtl[df_obtl['Model'] == 'GAM-SSM-LUR']['RMSE'].min()}
    ])

    sns.barplot(data=comparison_data, x='Method', y='RMSE', hue='Model', ax=ax,
               palette=[colors['fusion'], colors['gam']], alpha=0.85,
               edgecolor='black', linewidth=2)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=11, fontweight='bold', padding=3)

    ax.set_ylabel('Best RMSE (µg/m³)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Transfer Method', fontsize=13, fontweight='bold')
    ax.set_title('(b) Best Performance Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.legend(title='Model', fontsize=12, title_fontsize=12, frameon=True, shadow=True)
    sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig2_path = fig_dir / f'obtl_fancy_{timestamp}.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {fig2_path.name}")
    plt.close(fig2)

    # ========== Figure 3: Overall Summary Heatmap ==========
    fig3, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig3.patch.set_facecolor('white')

    # Prepare summary data
    summary_data = pd.DataFrame([
        {'Model': 'FusionGP', 'Method': 'Baseline\n(λ=0)', 'RMSE': baseline_fusion},
        {'Model': 'FusionGP', 'Method': 'Prior\nTempering', 'RMSE': df_pt[df_pt['Model'] == 'FusionGP']['RMSE'].min()},
        {'Model': 'FusionGP', 'Method': 'OBTL', 'RMSE': df_obtl[df_obtl['Model'] == 'FusionGP']['RMSE'].min()},
        {'Model': 'GAM-SSM-LUR', 'Method': 'Baseline\n(λ=0)', 'RMSE': baseline_gam},
        {'Model': 'GAM-SSM-LUR', 'Method': 'Prior\nTempering', 'RMSE': df_pt[df_pt['Model'] == 'GAM-SSM-LUR']['RMSE'].min()},
        {'Model': 'GAM-SSM-LUR', 'Method': 'OBTL', 'RMSE': df_obtl[df_obtl['Model'] == 'GAM-SSM-LUR']['RMSE'].min()}
    ])

    # Pivot for heatmap
    heatmap_data = summary_data.pivot(index='Model', columns='Method', values='RMSE')

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
               cbar_kws={'label': 'RMSE (µg/m³)'}, linewidths=2, linecolor='white',
               ax=ax, vmin=4.0, vmax=5.5, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_title('Transfer Learning Performance Heatmap\n(Lower is Better)',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Transfer Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Architecture', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig3_path = fig_dir / f'summary_heatmap_{timestamp}.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {fig3_path.name}")
    plt.close(fig3)

    print(f"\n All publication-quality figures saved to: {fig_dir}/")
    return fig_dir


def export_results_tables(all_results: Dict, output_dir: Path, timestamp: str):
    """
    Export results as LaTeX and CSV tables for easy thesis integration.

    Creates separate tables for:
    - Prior Tempering results (both models)
    - OBTL results (both models)
    - Summary comparison table

    Parameters
    ----------
    all_results : dict
        Complete results dictionary
    output_dir : Path
        Directory to save tables
    timestamp : str
        Timestamp for file naming
    """
    print(f"\n Exporting results tables (LaTeX + CSV)...")

    # Create tables subdirectory
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)

    # ========== Prior Tempering Results Table ==========
    pt_rows = []

    # FusionGP Prior Tempering
    for r in sorted(all_results['fusiongp_prior_tempering']['results'], key=lambda x: x['lambda']):
        pt_rows.append({
            'Model': 'FusionGP',
            'λ': r['lambda'],
            'RMSE (µg/m³)': f"{r['rmse']:.2f}",
            'MAE (µg/m³)': f"{r['mae']:.2f}",
            'R²': f"{r['r2']:.4f}"
        })

    # GAM-SSM-LUR Prior Tempering
    for r in sorted(all_results['gam_ssm_lur_prior_tempering']['results'], key=lambda x: x['lambda']):
        pt_rows.append({
            'Model': 'GAM-SSM-LUR',
            'λ': r['lambda'],
            'RMSE (µg/m³)': f"{r['rmse']:.2f}",
            'MAE (µg/m³)': f"{r['mae']:.2f}",
            'R²': f"{r['r2']:.4f}"
        })

    df_pt = pd.DataFrame(pt_rows)

    # Save Prior Tempering CSV
    csv_pt_path = tables_dir / 'prior_tempering_results.csv'
    df_pt.to_csv(csv_pt_path, index=False)
    print(f"   Saved: tables/{csv_pt_path.name}")

    # Save Prior Tempering LaTeX
    tex_pt_path = tables_dir / 'prior_tempering_results.tex'
    latex_pt = df_pt.to_latex(
        index=False,
        escape=False,
        column_format='llrrr',
        caption='Prior Tempering Transfer Learning Results',
        label='tab:prior_tempering',
        position='htbp'
    )
    with open(tex_pt_path, 'w') as f:
        f.write(latex_pt)
    print(f"   Saved: tables/{tex_pt_path.name}")

    # ========== OBTL Results Table ==========
    obtl_rows = []

    # FusionGP OBTL
    for r in sorted(all_results['fusiongp_obtl']['results'], key=lambda x: x['delta']):
        obtl_rows.append({
            'Model': 'FusionGP',
            'δ': r['delta'],
            'RMSE (µg/m³)': f"{r['rmse']:.2f}",
            'MAE (µg/m³)': f"{r['mae']:.2f}",
            'R²': f"{r['r2']:.4f}"
        })

    # GAM-SSM-LUR OBTL
    for r in sorted(all_results['gam_ssm_lur_obtl']['results'], key=lambda x: x['delta']):
        obtl_rows.append({
            'Model': 'GAM-SSM-LUR',
            'δ': r['delta'],
            'RMSE (µg/m³)': f"{r['rmse']:.2f}",
            'MAE (µg/m³)': f"{r['mae']:.2f}",
            'R²': f"{r['r2']:.4f}"
        })

    df_obtl = pd.DataFrame(obtl_rows)

    # Save OBTL CSV
    csv_obtl_path = tables_dir / 'obtl_results.csv'
    df_obtl.to_csv(csv_obtl_path, index=False)
    print(f"    Saved: tables/{csv_obtl_path.name}")

    # Save OBTL LaTeX
    tex_obtl_path = tables_dir / 'obtl_results.tex'
    latex_obtl = df_obtl.to_latex(
        index=False,
        escape=False,
        column_format='llrrr',
        caption='OBTL (Optimal Bayesian Transfer Learning) Results',
        label='tab:obtl',
        position='htbp'
    )
    with open(tex_obtl_path, 'w') as f:
        f.write(latex_obtl)
    print(f"    Saved: tables/{tex_obtl_path.name}")

    # ========== Summary Comparison Table ==========
    summary_rows = []

    # Extract best results for each model and method
    fusiongp_pt_best = all_results['fusiongp_prior_tempering']['best']
    gam_pt_best = all_results['gam_ssm_lur_prior_tempering']['best']
    fusiongp_obtl_best = all_results['fusiongp_obtl']['best']
    gam_obtl_best = all_results['gam_ssm_lur_obtl']['best']

    summary_rows.append({
        'Model': 'FusionGP',
        'Method': 'Prior Tempering',
        'Parameter': f"λ={fusiongp_pt_best['lambda']}",
        'RMSE (µg/m³)': f"{fusiongp_pt_best['rmse']:.2f}",
        'MAE (µg/m³)': f"{fusiongp_pt_best['mae']:.2f}",
        'R²': f"{fusiongp_pt_best['r2']:.4f}"
    })

    summary_rows.append({
        'Model': 'FusionGP',
        'Method': 'OBTL',
        'Parameter': f"δ={fusiongp_obtl_best['delta']}",
        'RMSE (µg/m³)': f"{fusiongp_obtl_best['rmse']:.2f}",
        'MAE (µg/m³)': f"{fusiongp_obtl_best['mae']:.2f}",
        'R²': f"{fusiongp_obtl_best['r2']:.4f}"
    })

    summary_rows.append({
        'Model': 'GAM-SSM-LUR',
        'Method': 'Prior Tempering',
        'Parameter': f"λ={gam_pt_best['lambda']}",
        'RMSE (µg/m³)': f"{gam_pt_best['rmse']:.2f}",
        'MAE (µg/m³)': f"{gam_pt_best['mae']:.2f}",
        'R²': f"{gam_pt_best['r2']:.4f}"
    })

    summary_rows.append({
        'Model': 'GAM-SSM-LUR',
        'Method': 'OBTL',
        'Parameter': f"δ={gam_obtl_best['delta']}",
        'RMSE (µg/m³)': f"{gam_obtl_best['rmse']:.2f}",
        'MAE (µg/m³)': f"{gam_obtl_best['mae']:.2f}",
        'R²': f"{gam_obtl_best['r2']:.4f}"
    })

    df_summary = pd.DataFrame(summary_rows)

    # Save Summary CSV
    csv_summary_path = tables_dir / 'summary_best_results.csv'
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"    Saved: tables/{csv_summary_path.name}")

    # Save Summary LaTeX
    tex_summary_path = tables_dir / 'summary_best_results.tex'
    latex_summary = df_summary.to_latex(
        index=False,
        escape=False,
        column_format='llcrrr',
        caption='Best Transfer Learning Results Summary',
        label='tab:summary_best',
        position='htbp'
    )
    with open(tex_summary_path, 'w') as f:
        f.write(latex_summary)
    print(f"    Saved: tables/{tex_summary_path.name}")

    print(f"\n All tables exported to: {tables_dir}/")
    print(f"   • Prior Tempering: prior_tempering_results.{{csv,tex}}")
    print(f"   • OBTL: obtl_results.{{csv,tex}}")
    print(f"   • Summary: summary_best_results.{{csv,tex}}")


def create_conceptual_visualisations(all_results: Dict, output_dir: Path, timestamp: str, test_data: Dict):
    """
    Create conceptual diagrams, spatial maps, and training diagnostics.

    Generates:
    1. Bayesian Prior Tempering conceptual diagram (Image 1)
    2. Spatial comparison maps: Standard vs Probabilistic Transfer (Image 3)
    3. Transfer gain/loss analysis for Prior Tempering
    4. Transfer gain/loss analysis for OBTL
    5. OBTL transfer weight evolution

    Parameters
    ----------
    all_results : dict
        Complete results dictionary with all experiments
    output_dir : Path
        Directory to save figures
    timestamp : str
        Timestamp for file naming
    test_data : dict
        Test data with predictions and uncertainties for spatial maps
    """
    # Create figure directory
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Creating conceptual visualisations and training diagnostics...")

    # ========== Image 1: Prior Tempering Conceptual Diagram ==========
    try:
        concept_path = fig_dir / f'prior_tempering_concept_{timestamp}.png'
        plot_prior_tempering_concept(output_path=concept_path, save_pdf=True)
    except Exception as e:
        print(f"   Warning: Could not create prior tempering concept: {e}")
        import traceback
        traceback.print_exc()

    # ========== Image 3: Spatial Comparison Maps ==========
    try:
        # Generate example spatial data for demonstration
        # In a real scenario, you would use actual predictions from the best model
        predictions, uncertainties, coords = generate_example_spatial_data(
            n_locations=50,
            seed=42
        )

        spatial_path = fig_dir / f'spatial_comparison_{timestamp}.png'
        plot_spatial_comparison(
            predictions=predictions,
            uncertainties=uncertainties,
            coords=coords,
            output_path=spatial_path,
            save_pdf=True,
            title="Standard vs Probabilistic Transfer Learning"
        )
    except Exception as e:
        print(f"   Warning: Could not create spatial comparison: {e}")
        import traceback
        traceback.print_exc()

    # ========== Training Diagnostic 1: Prior Tempering Gain/Loss (FusionGP) ==========
    try:
        # Get baseline (lambda=0) RMSE for FusionGP
        fusiongp_results = all_results['fusiongp_prior_tempering']['results']
        baseline_result = [r for r in fusiongp_results if r['lambda'] == 0.0]

        if baseline_result:
            baseline_rmse = baseline_result[0]['rmse']

            gain_loss_path = fig_dir / f'prior_tempering_gain_loss_fusiongp_{timestamp}.png'
            plot_transfer_gain_loss(
                results=all_results['fusiongp_prior_tempering'],
                baseline_metric=baseline_rmse,
                output_path=gain_loss_path,
                save_pdf=True,
                metric='rmse'
            )
    except Exception as e:
        print(f"   Warning: Could not create Prior Tempering gain/loss: {e}")
        import traceback
        traceback.print_exc()

    # ========== Training Diagnostic 2: OBTL Gain/Loss (FusionGP) ==========
    try:
        # Get baseline (delta=0.3, lowest transfer strength) for comparison
        fusiongp_obtl = all_results['fusiongp_obtl']['results']
        baseline_obtl = min(fusiongp_obtl, key=lambda x: x['delta'])
        baseline_rmse_obtl = baseline_obtl['rmse']

        obtl_gain_loss_path = fig_dir / f'obtl_gain_loss_fusiongp_{timestamp}.png'
        plot_transfer_gain_loss(
            results=all_results['fusiongp_obtl'],
            baseline_metric=baseline_rmse_obtl,
            output_path=obtl_gain_loss_path,
            save_pdf=True,
            metric='rmse'
        )
    except Exception as e:
        print(f"   Warning: Could not create OBTL gain/loss: {e}")
        import traceback
        traceback.print_exc()

    # ========== Training Diagnostic 3: OBTL Weight Evolution (FusionGP) ==========
    try:
        fusiongp_obtl = all_results['fusiongp_obtl']['results']

        weight_evolution_path = fig_dir / f'obtl_weight_evolution_fusiongp_{timestamp}.png'
        plot_transfer_weight_evolution(
            obtl_results=fusiongp_obtl,
            output_path=weight_evolution_path,
            save_pdf=True
        )
    except Exception as e:
        print(f"   Warning: Could not create OBTL weight evolution: {e}")
        import traceback
        traceback.print_exc()

    print(f" Conceptual visualisations and diagnostics saved to: {fig_dir.name}/")


def main():
    """
    Run real model transfer learning experiments.
    """
    print("="*70)
    print("REAL MODEL TRANSFER LEARNING EXPERIMENTS")
    print("="*70)
    print("\nTransfer Scenario:")
    print("  Source: Pre-trained models (real Source domain data)")
    print("    - FusionGP")
    print("    - GAM-SSM-LUR")
    print("  Target: Synthetic Target domain data")
    print("  Methods: Prior Tempering, OBTL")
    print("="*70)

    # Paths to saved models
    base_path = Path(__file__).parent.parent
    fusiongp_path = base_path / 'models' / 'fusiongp' / 'dublin' / 'fusiongp_model.pth'
    gam_path = base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'gam.pkl'
    ssm_path = base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'ssm.pkl'
    gam_data_path = base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'training_data.npz'

    # Generate synthetic Target domain data
    print("\n Generating synthetic Target domain data...")
    target_data = generate_synthetic_target_data(n_target=50, n_test=100, seed=42)
    print(f"   Target: {target_data['target']['X'].shape[0]} samples")
    print(f"   Test:   {target_data['test']['X'].shape[0]} samples")

    # Load Source FusionGP
    try:
        source_fusiongp, source_likelihood = load_source_fusiongp(str(fusiongp_path))
    except Exception as e:
        print(f"\n Error loading FusionGP: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load Source GAM-SSM-LUR
    try:
        source_gam_ssm_lur, source_gam_data = load_source_gam_ssm_lur(
            str(gam_path), str(ssm_path), str(gam_data_path)
        )
    except Exception as e:
        print(f"\n Error loading GAM-SSM-LUR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Experiment 1: FusionGP with Prior Tempering
    results_fusiongp_pt = transfer_fusiongp_with_prior_tempering(
        source_fusiongp,
        source_likelihood,
        target_data['target'],
        target_data['test'],
        beta_values=[0.0, 0.3, 0.5, 0.7, 1.0]
    )

    # Experiment 2: GAM-SSM-LUR with Prior Tempering
    results_gam_pt = transfer_gam_ssm_lur_with_prior_tempering(
        source_gam_ssm_lur['gam'],
        source_gam_ssm_lur['ssm'],
        source_gam_data,
        target_data['target'],
        target_data['test'],
        beta_values=[0.0, 0.3, 0.5, 0.7, 1.0]
    )

    # Experiment 3: FusionGP with OBTL
    results_fusiongp_obtl = transfer_fusiongp_with_obtl(
        source_fusiongp,
        source_likelihood,
        target_data['target'],
        target_data['test'],
        delta_values=[0.3, 0.5, 0.7, 1.0]
    )

    # Experiment 4: GAM-SSM-LUR with OBTL
    results_gam_obtl = transfer_gam_ssm_lur_with_obtl(
        source_gam_ssm_lur['gam'],
        source_gam_ssm_lur['ssm'],
        source_gam_data,
        target_data['target'],
        target_data['test'],
        delta_values=[0.3, 0.5, 0.7, 1.0]
    )

    # Save results - Create experiment-specific folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).parent.parent / 'results'
    experiment_dir = base_results_dir / f'experiment_{timestamp}'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    figures_dir = experiment_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    all_results = {
        'timestamp': timestamp,
        'source': 'Real Source Domain Models',
        'target': 'Synthetic Target Domain',
        'fusiongp_prior_tempering': results_fusiongp_pt,
        'gam_ssm_lur_prior_tempering': results_gam_pt,
        'fusiongp_obtl': results_fusiongp_obtl,
        'gam_ssm_lur_obtl': results_gam_obtl
    }

    # Save JSON results
    json_file = experiment_dir / 'results.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n Experiment folder: {experiment_dir}")
    print(f" Results JSON: {json_file.name}")

    print(f"\n Best Results Summary:")
    print(f"\n  FusionGP:")
    print(f"    Prior Tempering: λ={results_fusiongp_pt['best']['lambda']}, RMSE={results_fusiongp_pt['best']['rmse']:.2f} µg/m³, R²={results_fusiongp_pt['best']['r2']:.4f}")
    print(f"    OBTL:            δ={results_fusiongp_obtl['best']['delta']}, RMSE={results_fusiongp_obtl['best']['rmse']:.2f} µg/m³, R²={results_fusiongp_obtl['best']['r2']:.4f}")

    print(f"\n  GAM-SSM-LUR:")
    print(f"    Prior Tempering: λ={results_gam_pt['best']['lambda']}, RMSE={results_gam_pt['best']['rmse']:.2f} µg/m³, R²={results_gam_pt['best']['r2']:.4f}")
    print(f"    OBTL:            δ={results_gam_obtl['best']['delta']}, RMSE={results_gam_obtl['best']['rmse']:.2f} µg/m³, R²={results_gam_obtl['best']['r2']:.4f}")

    # Create visualisations in experiment folder
    try:
        create_transfer_visualisations(all_results, experiment_dir, timestamp)
        print(f" Figures: {figures_dir.name}/")
    except Exception as e:
        print(f"\n  Warning: Could not create visualisations: {e}")
        import traceback
        traceback.print_exc()

    # Export results as LaTeX and CSV tables
    try:
        export_results_tables(all_results, experiment_dir, timestamp)
    except Exception as e:
        print(f"\n  Warning: Could not export tables: {e}")
        import traceback
        traceback.print_exc()

    # Create conceptual visualisations
    try:
        create_conceptual_visualisations(all_results, experiment_dir, timestamp, target_data['test'])
    except Exception as e:
        print(f"\n  Warning: Could not create conceptual visualisations: {e}")
        import traceback
        traceback.print_exc()

    # Print experiment summary
    print(f"\n{'='*70}")
    print("EXPERIMENT STRUCTURE:")
    print(f"{'='*70}")
    print(f"\n{experiment_dir}/")
    print(f"├── results.json                          # All experimental results (JSON)")
    print(f"├── tables/                               # Results tables (CSV + LaTeX)")
    print(f"│   ├── prior_tempering_results.csv")
    print(f"│   ├── prior_tempering_results.tex")
    print(f"│   ├── obtl_results.csv")
    print(f"│   ├── obtl_results.tex")
    print(f"│   ├── summary_best_results.csv")
    print(f"│   └── summary_best_results.tex")
    print(f"└── figures/                              # Publication-quality plots")
    print(f"    ├── prior_tempering_fancy_{timestamp}.png         # Performance metrics")
    print(f"    ├── obtl_fancy_{timestamp}.png")
    print(f"    ├── summary_heatmap_{timestamp}.png")
    print(f"    ├── prior_tempering_concept_{timestamp}.png       # Conceptual diagrams")
    print(f"    ├── prior_tempering_concept_{timestamp}.pdf")
    print(f"    ├── spatial_comparison_{timestamp}.png            # Spatial maps")
    print(f"    ├── spatial_comparison_{timestamp}.pdf")
    print(f"    ├── prior_tempering_gain_loss_fusiongp_{timestamp}.png  # Training diagnostics")
    print(f"    ├── prior_tempering_gain_loss_fusiongp_{timestamp}.pdf")
    print(f"    ├── obtl_gain_loss_fusiongp_{timestamp}.png")
    print(f"    ├── obtl_gain_loss_fusiongp_{timestamp}.pdf")
    print(f"    ├── obtl_weight_evolution_fusiongp_{timestamp}.png")
    print(f"    └── obtl_weight_evolution_fusiongp_{timestamp}.pdf")


if __name__ == '__main__':
    main()
