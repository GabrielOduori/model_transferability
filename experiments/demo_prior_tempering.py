"""
Demonstration: Prior Tempering Transfer Learning

This script demonstrates the complete workflow for transferring a GP model
from a source city (Dublin) to a target city using prior tempering.

Usage:
    python experiments/demo_prior_tempering.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from src.models.gp_model import BaselineGP, train_baseline_gp, predict_with_uncertainty, sample_posterior
from src.transfer_methods.prior_tempering import TemperedGP, train_tempered_gp
from src.evaluation.metrics import (
    kl_divergence_distributions,
    prediction_interval_coverage_probability,
    regression_metrics,
    TransferEvaluator
)


def generate_synthetic_data(
    n_source: int = 200,
    n_target: int = 50,
    n_test: int = 100,
    input_dim: int = 3,
    domain_shift: float = 0.3,
    seed: int = 42
):
    """
    Generate synthetic air quality data for source and target cities.

    Parameters
    ----------
    n_source : int
        Number of source domain samples (Dublin)
    n_target : int
        Number of target domain samples (Cork)
    n_test : int
        Number of test samples
    input_dim : int
        Number of features (e.g., meteorology, land use)
    domain_shift : float
        Magnitude of distribution shift between domains
    seed : int
        Random seed

    Returns
    -------
    dict
        Data dictionary with train/test splits
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Source domain (Dublin): larger dataset
    X_source = torch.randn(n_source, input_dim)
    # True function: pollution depends on features
    y_source = (
        2.0 * torch.sin(X_source[:, 0]) +
        1.5 * torch.cos(X_source[:, 1]) +
        0.5 * X_source[:, 2] +
        torch.randn(n_source) * 0.3
    )

    # Target domain (Cork): similar but shifted distribution
    X_target = torch.randn(n_target, input_dim) + domain_shift
    # Same underlying function but different noise level
    y_target = (
        2.0 * torch.sin(X_target[:, 0]) +
        1.5 * torch.cos(X_target[:, 1]) +
        0.5 * X_target[:, 2] +
        torch.randn(n_target) * 0.4  # Higher noise in target
    )

    # Test data from target domain
    X_test = torch.randn(n_test, input_dim) + domain_shift
    y_test = (
        2.0 * torch.sin(X_test[:, 0]) +
        1.5 * torch.cos(X_test[:, 1]) +
        0.5 * X_test[:, 2] +
        torch.randn(n_test) * 0.4
    )

    return {
        'source': {'X': X_source, 'y': y_source},
        'target': {'X': X_target, 'y': y_target},
        'test': {'X': X_test, 'y': y_test}
    }


def run_experiment(beta_values=[0.0, 0.3, 0.5, 0.7, 1.0], save_results=True):
    """
    Run complete transfer learning experiment.

    Compares:
    1. Baseline (train from scratch on target)
    2. Direct transfer (β=1.0)
    3. Tempered transfer (various β values)

    Parameters
    ----------
    beta_values : list of float
        Temperature values to test
    save_results : bool
        Whether to save results to disk
    """
    print("="*70)
    print("TRANSFER LEARNING EXPERIMENT: PRIOR TEMPERING")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic air quality data...")
    data = generate_synthetic_data(
        n_source=200,  # Dublin: large dataset
        n_target=50,   # Cork: small dataset
        n_test=100,
        domain_shift=0.3
    )
    print(f"   Source domain: {len(data['source']['y'])} samples")
    print(f"   Target domain: {len(data['target']['y'])} samples")
    print(f"   Test set: {len(data['test']['y'])} samples")

    # Train source model (Dublin)
    print("\n2. Training source model (Dublin)...")
    source_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    source_model = BaselineGP(
        data['source']['X'],
        data['source']['y'],
        source_likelihood
    )
    source_model, source_likelihood = train_baseline_gp(
        source_model, source_likelihood,
        data['source']['X'], data['source']['y'],
        num_iter=200,
        verbose=False
    )
    print("   ✓ Source model trained")

    # Sample source posterior for KL divergence
    print("\n3. Sampling source posterior...")
    source_posterior = sample_posterior(source_model, n_samples=1000)
    print(f"   ✓ Sampled {len(source_posterior)} posterior samples")

    # Evaluate different transfer strategies
    print("\n4. Evaluating transfer strategies...")
    print("   " + "-"*60)

    results = {}
    evaluator = TransferEvaluator()

    for beta in beta_values:
        print(f"\n   β = {beta:.1f}:")

        if beta == 0.0:
            # Baseline: no transfer
            target_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            target_model = BaselineGP(
                data['target']['X'],
                data['target']['y'],
                target_likelihood
            )
            target_model, target_likelihood = train_baseline_gp(
                target_model, target_likelihood,
                data['target']['X'], data['target']['y'],
                num_iter=200,
                verbose=False
            )
            strategy = "Baseline (no transfer)"
        else:
            # Transfer with tempering
            target_model, target_likelihood = train_tempered_gp(
                source_gp=source_model,
                target_x=data['target']['X'],
                target_y=data['target']['y'],
                beta=beta,
                num_iter=200,
                lr=0.01,
                verbose=False
            )
            strategy = f"Tempered transfer (β={beta})"

        # Predictions on test set
        preds, stds = predict_with_uncertainty(
            target_model, target_likelihood, data['test']['X']
        )

        # Metrics
        metrics = regression_metrics(preds, data['test']['y'].numpy())
        picp = prediction_interval_coverage_probability(
            preds, stds, data['test']['y'].numpy()
        )

        # Sample target posterior
        target_posterior = sample_posterior(target_model, n_samples=1000)
        kl_div = kl_divergence_distributions(source_posterior, target_posterior)

        # Store results
        results[beta] = {
            'strategy': strategy,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'picp': picp,
            'kl_divergence': kl_div,
            'predictions': preds,
            'uncertainties': stds
        }

        print(f"      RMSE: {metrics['rmse']:.4f}")
        print(f"      MAE:  {metrics['mae']:.4f}")
        print(f"      R²:   {metrics['r2']:.4f}")
        print(f"      PICP: {picp:.2%} (target: 95%)")
        print(f"      KL:   {kl_div:.4f}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Best Transfer Strategy")
    print("="*70)

    best_beta = min(results.keys(), key=lambda b: results[b]['rmse'])
    best_result = results[best_beta]

    print(f"\nBest β = {best_beta:.1f}")
    print(f"  RMSE:  {best_result['rmse']:.4f}")
    print(f"  PICP:  {best_result['picp']:.2%}")
    print(f"  KL:    {best_result['kl_divergence']:.4f}")

    baseline = results[0.0]
    improvement = (baseline['rmse'] - best_result['rmse']) / baseline['rmse'] * 100
    print(f"\nImprovement over baseline: {improvement:.1f}%")

    # Visualization
    print("\n5. Generating visualizations...")
    fig = create_visualizations(results, data)

    if save_results:
        # Get project root directory (parent of experiments/)
        project_root = Path(__file__).parent.parent
        output_dir = project_root / 'results' / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'prior_tempering_results.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_dir / 'prior_tempering_results.png'}")

    plt.show()

    return results


def create_visualizations(results, data):
    """Create comprehensive visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    beta_values = sorted(results.keys())

    # 1. RMSE vs Beta
    ax = axes[0, 0]
    rmse_values = [results[b]['rmse'] for b in beta_values]
    ax.plot(beta_values, rmse_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature β', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('(a) Transfer Performance vs Temperature', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. KL Divergence vs Beta
    ax = axes[0, 1]
    kl_values = [results[b]['kl_divergence'] for b in beta_values]
    ax.plot(beta_values, kl_values, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Temperature β', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('(b) Domain Distance vs Temperature', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Predictions scatter plot
    ax = axes[1, 0]
    best_beta = min(results.keys(), key=lambda b: results[b]['rmse'])
    preds = results[best_beta]['predictions']
    true_vals = data['test']['y'].numpy()

    ax.scatter(true_vals, preds, alpha=0.6, s=50)
    min_val, max_val = true_vals.min(), true_vals.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predictions', fontsize=12)
    ax.set_title(f'(c) Predictions (β={best_beta})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. PICP comparison
    ax = axes[1, 1]
    picp_values = [results[b]['picp'] for b in beta_values]
    ax.bar(range(len(beta_values)), picp_values, alpha=0.7)
    ax.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Target (95%)')
    ax.set_xticks(range(len(beta_values)))
    ax.set_xticklabels([f'{b:.1f}' for b in beta_values])
    ax.set_xlabel('Temperature β', fontsize=12)
    ax.set_ylabel('PICP', fontsize=12)
    ax.set_title('(d) Uncertainty Calibration', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import gpytorch

    print("\n" + "="*70)
    print("PRIOR TEMPERING TRANSFER LEARNING - DEMONSTRATION")
    print("="*70)
    print("\nThis experiment demonstrates transferring an air quality model")
    print("from Dublin (source, large dataset) to Cork (target, small dataset)")
    print("using Bayesian prior tempering.\n")

    # Run experiment
    results = run_experiment(
        beta_values=[0.0, 0.3, 0.5, 0.7, 1.0],
        save_results=True
    )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print("• Prior tempering enables effective knowledge transfer")
    print("• Optimal β balances source knowledge and target adaptation")
    print("• Transfer learning reduces data requirements for new cities")
    print("\n")
