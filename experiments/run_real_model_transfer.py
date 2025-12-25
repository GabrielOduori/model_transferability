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


def generate_synthetic_cork_data(n_target=50, n_test=100, seed=42):
    """
    Generate synthetic Cork data for transfer learning experiments.

    Simulates Cork air quality measurements with domain shift from Dublin.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Target domain (Cork): shifted distribution from Dublin
    X_target = torch.randn(n_target, 10) * 1.5 + 0.5
    y_target = (
        2.0 * X_target[:, 0] +
        1.5 * X_target[:, 1] -
        0.8 * X_target[:, 2] +
        torch.randn(n_target) * 0.5 +
        1.0  # Systematic offset from Dublin
    ) * 10.0 + 25.0  # Scale to realistic NO₂ values (µg/m³)

    # Test data (same distribution as Cork target)
    X_test = torch.randn(n_test, 10) * 1.5 + 0.5
    y_test = (
        2.0 * X_test[:, 0] +
        1.5 * X_test[:, 1] -
        0.8 * X_test[:, 2] +
        torch.randn(n_test) * 0.5 +
        1.0
    ) * 10.0 + 25.0

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
        Path to saved FusionGP model (.pt file)

    Returns
    -------
    model : gpytorch.models.ExactGP
        Loaded FusionGP model
    likelihood : gpytorch.likelihoods.Likelihood
        Associated likelihood
    """
    print(f"\n📦 Loading Dublin FusionGP model from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract model components from checkpoint
    # Note: Adjust this based on your actual checkpoint structure
    if isinstance(checkpoint, dict):
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        # You'll need to reconstruct the model from the checkpoint
        # This is a placeholder - adjust based on your actual model structure
        raise NotImplementedError(
            "Please implement FusionGP model loading based on your checkpoint structure. "
            "Check the checkpoint keys and model architecture."
        )
    else:
        model = checkpoint  # If checkpoint is the model directly
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model.eval()
    print("   ✓ FusionGP loaded successfully")

    return model, likelihood


def load_dublin_gam_ssm_lur(model_path: str = None):
    """
    Load pre-trained Dublin GAM-SSM-LUR model.

    Parameters
    ----------
    model_path : str, optional
        Path to saved GAM-SSM-LUR model

    Returns
    -------
    model :
        Loaded GAM-SSM-LUR model
    """
    print(f"\n📦 Loading Dublin GAM-SSM-LUR model")

    if model_path is None:
        print("   ⚠️  No GAM-SSM-LUR checkpoint provided")
        print("   Using synthetic model for now")
        # Create placeholder - replace with actual loading
        raise NotImplementedError(
            "Please provide GAM-SSM-LUR model checkpoint path or implement loading."
        )

    # Load GAM-SSM-LUR model
    # This will depend on how you saved it
    raise NotImplementedError("Implement GAM-SSM-LUR loading based on your save format")


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

    # Paths to your saved models
    fusiongp_path = Path(__file__).parent.parent / \
        'gam_ssm_lur' / 'fusionGP2' / 'fusiongp' / 'notebooks' / 'fusiongp_model.pt'

    # Generate synthetic Cork data
    print("\n📦 Generating synthetic Cork data...")
    cork_data = generate_synthetic_cork_data(n_target=50, n_test=100, seed=42)
    print(f"   Target: {cork_data['target']['X'].shape[0]} samples")
    print(f"   Test:   {cork_data['test']['X'].shape[0]} samples")

    # Load Dublin FusionGP
    try:
        dublin_fusiongp, dublin_likelihood = load_dublin_fusiongp(str(fusiongp_path))
    except NotImplementedError as e:
        print(f"\n⚠️  {e}")
        print("\nNEXT STEPS:")
        print("1. Inspect your saved model checkpoint:")
        print(f"   checkpoint = torch.load('{fusiongp_path}')")
        print("   print(checkpoint.keys())")
        print("2. Update load_dublin_fusiongp() to reconstruct your model")
        print("3. Provide Dublin training data for OBTL experiments")
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
