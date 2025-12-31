"""
Run Transfer Learning Experiments Using Saved Synthetic Data

This script demonstrates how to use the saved synthetic target data
(data/synthetic_target/target_data_seed42.npz) to reproduce experimental results.

Usage:
    python experiments/run_transfer_with_saved_data.py
"""

import numpy as np
import torch
import sys
from pathlib import Path
from datetime import datetime
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gp_model import BaselineGP, predict_with_uncertainty
from src.evaluation.metrics import regression_metrics
import gpytorch


def load_saved_synthetic_data():
    """
    Load saved synthetic target domain data.

    Returns
    -------
    dict
        Dictionary with 'target' and 'test' data
    """
    data_file = Path(__file__).parent.parent / 'data' / 'synthetic_target' / 'target_data_seed42.npz'

    if not data_file.exists():
        raise FileNotFoundError(
            f"Synthetic data not found: {data_file}\n"
            "Generate it by running: python experiments/run_real_model_transfer.py"
        )

    print(f"Loading saved synthetic data: {data_file}")
    data = np.load(data_file)

    # Print metadata
    print(f"  Metadata:")
    print(f"    Seed: {data['seed']}")
    print(f"    Training samples: {data['n_target']}")
    print(f"    Test samples: {data['n_test']}")
    print(f"    Features: {data['n_features']}")
    print(f"    Domain shift: {data['domain_shift']}")
    print(f"    Noise std: {data['noise_std']} µg/m³")

    # Convert to torch tensors
    X_target = torch.from_numpy(data['X_target'])
    y_target = torch.from_numpy(data['y_target'])
    X_test = torch.from_numpy(data['X_test'])
    y_test = torch.from_numpy(data['y_test'])

    print(f"\n✓ Loaded data:")
    print(f"    X_target: {X_target.shape}")
    print(f"    y_target: {y_target.shape}")
    print(f"    X_test: {X_test.shape}")
    print(f"    y_test: {y_test.shape}")

    return {
        'target': {'X': X_target, 'y': y_target},
        'test': {'X': X_test, 'y': y_test},
        'metadata': {
            'seed': int(data['seed']),
            'n_target': int(data['n_target']),
            'n_test': int(data['n_test']),
            'domain_shift': float(data['domain_shift']),
            'noise_std': float(data['noise_std'])
        }
    }


def train_baseline_gp(X_train, y_train, X_test, y_test):
    """
    Train baseline GP (no transfer) on target data.

    This simulates λ=0.0 (pure target learning).
    """
    print("\n" + "="*60)
    print("Training Baseline GP (No Transfer, λ=0.0)")
    print("="*60)

    # Create GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BaselineGP(X_train, y_train, likelihood)

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train
    n_iter = 100
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"  Iter {i+1}/{n_iter}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predictions = likelihood(model(X_test))
        mean = predictions.mean
        stddev = predictions.stddev

    # Metrics
    metrics = regression_metrics(
        true_values=y_test.numpy(),
        predictions=mean.numpy()
    )

    print(f"\n✓ Baseline Results:")
    print(f"    RMSE: {metrics['rmse']:.3f} µg/m³")
    print(f"    MAE:  {metrics['mae']:.3f} µg/m³")
    print(f"    R²:   {metrics['r2']:.3f}")

    return {
        'lambda': 0.0,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2']
    }


def simulate_prior_tempering(X_train, y_train, X_test, y_test, lambda_val):
    """
    Simulate Prior Tempering transfer learning.

    NOTE: This is a simplified simulation without actual source models.
    For real transfer, you need pre-trained source models.

    Parameters
    ----------
    lambda_val : float
        Temperature parameter (0.0 = no transfer, 1.0 = full transfer)
    """
    print(f"\nTraining GP with Prior Tempering (λ={lambda_val})")

    # Create GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BaselineGP(X_train, y_train, likelihood)

    # Simulate source hyperparameters (from thesis: GAM-SSM-LUR)
    # Real implementation would load these from source model
    source_noise_var = 0.0157  # From thesis
    source_lengthscale = 0.68  # From thesis

    # Apply tempering: blend source and target priors
    if lambda_val > 0:
        # Tempered noise variance
        target_noise_var = 1.5 ** 2  # Default uninformative
        tempered_noise_var = (source_noise_var ** lambda_val) * (target_noise_var ** (1 - lambda_val))

        # Set as prior
        model.likelihood.noise = tempered_noise_var
        model.covar_module.base_kernel.lengthscale = source_lengthscale * lambda_val + 1.0 * (1 - lambda_val)

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train (fewer iterations for tempered prior)
    n_iter = 50
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predictions = likelihood(model(X_test))
        mean = predictions.mean

    # Metrics
    metrics = regression_metrics(
        true_values=y_test.numpy(),
        predictions=mean.numpy()
    )

    print(f"  λ={lambda_val}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")

    return {
        'lambda': lambda_val,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2']
    }


def main():
    """Main experimental pipeline."""
    print("="*60)
    print("Transfer Learning with Saved Synthetic Data")
    print("="*60)

    # Load saved data
    data = load_saved_synthetic_data()

    X_train = data['target']['X']
    y_train = data['target']['y']
    X_test = data['test']['X']
    y_test = data['test']['y']

    # Run baseline (no transfer)
    baseline_result = train_baseline_gp(X_train, y_train, X_test, y_test)

    # Run Prior Tempering with different λ values
    print("\n" + "="*60)
    print("Prior Tempering Transfer (Simulated)")
    print("="*60)
    print("NOTE: This is simplified simulation without actual source models.")
    print("For real results, run: python experiments/run_real_model_transfer.py")

    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = []

    for lam in lambda_values:
        result = simulate_prior_tempering(X_train, y_train, X_test, y_test, lam)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"{'λ':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Improvement':>12}")
    print("-"*60)

    for result in results:
        improvement = (baseline_result['rmse'] - result['rmse']) / baseline_result['rmse'] * 100
        print(f"{result['lambda']:<8.1f} {result['rmse']:>10.3f} {result['mae']:>10.3f} "
              f"{result['r2']:>10.3f} {improvement:>11.2f}%")

    # Save results
    output_dir = Path('results/synthetic_data_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'simulated_transfer_{timestamp}.json'

    output_data = {
        'timestamp': timestamp,
        'data_file': 'data/synthetic_target/target_data_seed42.npz',
        'metadata': data['metadata'],
        'baseline': baseline_result,
        'prior_tempering_simulated': results,
        'note': 'Simulated transfer without actual source models'
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved: {output_file}")

    # Compare with thesis results
    print("\n" + "="*60)
    print("Comparison with Thesis Results")
    print("="*60)
    print("Expected (from thesis with real source models):")
    print("  GAM-SSM-LUR λ=0.3: RMSE=4.39 µg/m³, Improvement=+9.82%")
    print("  FusionGP λ=0.0:    RMSE=4.86 µg/m³, Improvement=0.00%")
    print("\nYour simulated results will differ because:")
    print("  1. Real source models not used (GAM-SSM-LUR, FusionGP)")
    print("  2. Simplified tempering simulation")
    print("  3. Need actual pre-trained hyperparameters")
    print("\nTo get exact thesis results:")
    print("  python experiments/run_real_model_transfer.py")
    print("  (Requires source models in models/ directory)")


if __name__ == '__main__':
    main()
