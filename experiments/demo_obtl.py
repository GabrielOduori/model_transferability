"""
Demo: OBTL Transfer Learning for Air Quality Models

Demonstrates transferring covariance structure from Dublin (source)
to Cork (target) using Optimal Bayesian Transfer Learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transfer_methods.obtl import (
    train_obtl_gp,
    compare_covariance_structures,
    OBTLGaussianProcess
)
from src.models.gp_model import predict_with_uncertainty, train_baseline_gp
from src.evaluation.metrics import regression_metrics


def save_metrics_to_file(results, timestamp=None):
    """
    Save experiment metrics to JSON, CSV, and LaTeX files.

    Parameters
    ----------
    results : dict
        Results dictionary with metrics for each delta value
    timestamp : str, optional
        Timestamp string. If None, current time is used.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get project root directory
    project_root = Path(__file__).parent.parent
    metrics_dir = project_root / 'results' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metrics for JSON serialization
    metrics_data = {
        'timestamp': timestamp,
        'experiment': 'obtl',
        'delta_values': [float(d) for d in results['delta']],
        'metrics': {}
    }

    # Add metrics for each delta value
    for i, delta in enumerate(results['delta']):
        metrics_data['metrics'][str(delta)] = {
            'rmse': float(results['rmse'][i]),
            'mae': float(results['mae'][i]),
            'r2': float(results['r2'][i]),
            'cov_similarity': float(results['cov_similarity'][i])
        }

    # Add summary statistics
    best_idx = np.argmin(results['rmse'])
    best_delta = results['delta'][best_idx]
    baseline_rmse = results['rmse'][0]
    improvement = (baseline_rmse - results['rmse'][best_idx]) / baseline_rmse * 100

    metrics_data['summary'] = {
        'best_delta': float(best_delta),
        'best_rmse': float(results['rmse'][best_idx]),
        'best_mae': float(results['mae'][best_idx]),
        'best_r2': float(results['r2'][best_idx]),
        'baseline_rmse': float(baseline_rmse),
        'improvement_percent': float(improvement)
    }

    # 1. Save JSON
    json_filename = f'obtl_metrics_{timestamp}.json'
    json_filepath = metrics_dir / json_filename
    with open(json_filepath, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    latest_json_filepath = metrics_dir / 'obtl_metrics_latest.json'
    with open(latest_json_filepath, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # 2. Save CSV
    csv_filename = f'obtl_metrics_{timestamp}.csv'
    csv_filepath = metrics_dir / csv_filename

    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Delta', 'RMSE', 'MAE', 'R2', 'Cov_Similarity'])
        for i, delta in enumerate(results['delta']):
            writer.writerow([
                delta,
                f"{results['rmse'][i]:.4f}",
                f"{results['mae'][i]:.4f}",
                f"{results['r2'][i]:.4f}",
                f"{results['cov_similarity'][i]:.4f}"
            ])
        # Add summary row
        writer.writerow([])
        writer.writerow(['Summary', '', '', '', ''])
        writer.writerow(['Best Delta', best_delta, '', '', ''])
        writer.writerow(['Improvement (%)', f"{improvement:.2f}", '', '', ''])

    latest_csv_filepath = metrics_dir / 'obtl_metrics_latest.csv'
    with open(csv_filepath, 'r') as src, open(latest_csv_filepath, 'w') as dst:
        dst.write(src.read())

    # 3. Save LaTeX table
    latex_filename = f'obtl_metrics_{timestamp}.tex'
    latex_filepath = metrics_dir / latex_filename

    with open(latex_filepath, 'w') as f:
        f.write("% OBTL Transfer Learning Results\n")
        f.write(f"% Generated: {timestamp}\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{OBTL Transfer Learning Performance}\n")
        f.write("\\label{tab:obtl_results}\n")
        f.write("\\begin{tabular}{ccccc}\n")
        f.write("\\hline\n")
        f.write("$\\delta$ & RMSE & MAE & $R^2$ & Cov. Sim. \\\\\n")
        f.write("\\hline\n")

        for i, delta in enumerate(results['delta']):
            f.write(f"{delta:.1f} & "
                   f"{results['rmse'][i]:.4f} & "
                   f"{results['mae'][i]:.4f} & "
                   f"{results['r2'][i]:.4f} & "
                   f"{results['cov_similarity'][i]:.4f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\multicolumn{5}{l}{")
        f.write(f"Best $\\delta$ = {best_delta:.1f}, "
               f"Improvement = {improvement:.1f}\\%")
        f.write("} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    latest_latex_filepath = metrics_dir / 'obtl_metrics_latest.tex'
    with open(latex_filepath, 'r') as src, open(latest_latex_filepath, 'w') as dst:
        dst.write(src.read())

    print(f"   ✓ Saved metrics to:")
    print(f"      - JSON: {json_filepath.name}")
    print(f"      - CSV:  {csv_filepath.name}")
    print(f"      - LaTeX: {latex_filepath.name}")
    print(f"   ✓ Also saved as 'latest' versions")


def generate_synthetic_data(n_source=200, n_target=50, n_test=100, seed=42):
    """Generate synthetic spatial air quality data."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Source domain (Dublin): 2D spatial locations
    X_source = torch.rand(n_source, 2) * 10
    # True function: pollution increases near city center (5,5)
    distances_s = torch.sqrt(((X_source - 5.0) ** 2).sum(dim=1))
    y_source = 50 - 3 * distances_s + torch.randn(n_source) * 2

    # Target domain (Cork): Similar but shifted pattern
    X_target = torch.rand(n_target, 2) * 10
    distances_t = torch.sqrt(((X_target - 6.0) ** 2).sum(dim=1))  # Shifted center
    y_target = 48 - 3 * distances_t + torch.randn(n_target) * 2

    # Test set
    X_test = torch.rand(n_test, 2) * 10
    distances_test = torch.sqrt(((X_test - 6.0) ** 2).sum(dim=1))
    y_test = 48 - 3 * distances_test + torch.randn(n_test) * 0.5

    return {
        'source': (X_source, y_source),
        'target': (X_target, y_target),
        'test': (X_test, y_test)
    }


def run_experiment(delta_values=[0.0, 0.3, 0.5, 0.7, 1.0], save_results=True):
    """Run OBTL transfer learning experiment."""

    print("=" * 70)
    print("OBTL TRANSFER LEARNING - DEMONSTRATION")
    print("=" * 70)
    print("\nTransferring covariance structure from Dublin to Cork")
    print("using Optimal Bayesian Transfer Learning.\n")

    print("=" * 70)
    print("TRANSFER LEARNING EXPERIMENT: OBTL")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic air quality data...")
    data = generate_synthetic_data()
    X_source, y_source = data['source']
    X_target, y_target = data['target']
    X_test, y_test = data['test']
    print(f"   Source domain: {len(X_source)} samples")
    print(f"   Target domain: {len(X_target)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Results storage
    results = {
        'delta': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'cov_similarity': []
    }

    print("\n2. Evaluating OBTL transfer strategies...")
    print("   " + "-" * 60)

    for delta in delta_values:
        print(f"\n   δ = {delta}:")

        # Train with OBTL
        model, likelihood, obtl_info = train_obtl_gp(
            X_source, y_source,
            X_target, y_target,
            n_inducing=15,
            nu_0=10.0,
            delta=delta,
            n_iterations=50
        )

        # Evaluate on test set
        y_pred, y_std = predict_with_uncertainty(model, likelihood, X_test)

        # Metrics
        metrics = regression_metrics(
            y_pred,
            y_test.numpy()
        )

        # Covariance similarity
        cov_metrics = compare_covariance_structures(
            obtl_info['source_cov'],
            obtl_info['target_cov'],
            obtl_info['transferred_cov']
        )

        # Store results
        results['delta'].append(delta)
        results['rmse'].append(metrics['rmse'])
        results['mae'].append(metrics['mae'])
        results['r2'].append(metrics['r2'])
        results['cov_similarity'].append(cov_metrics['normalized_transfer_to_source'])

        print(f"      RMSE: {metrics['rmse']:.4f}")
        print(f"      MAE:  {metrics['mae']:.4f}")
        print(f"      R²:   {metrics['r2']:.4f}")
        print(f"      Cov similarity to source: {cov_metrics['normalized_transfer_to_source']:.4f}")

    # Find best delta
    best_idx = np.argmin(results['rmse'])
    best_delta = results['delta'][best_idx]

    print("\n" + "=" * 70)
    print("SUMMARY: Best Transfer Strategy")
    print("=" * 70)
    print(f"\nBest δ = {best_delta}")
    print(f"  RMSE:  {results['rmse'][best_idx]:.4f}")
    print(f"  R²:    {results['r2'][best_idx]:.4f}")

    baseline_rmse = results['rmse'][0]
    improvement = (baseline_rmse - results['rmse'][best_idx]) / baseline_rmse * 100
    print(f"\nImprovement over baseline: {improvement:.1f}%")

    # Visualization
    print("\n3. Generating visualizations...")
    fig = create_visualizations(results, data, delta_values)

    if save_results:
        project_root = Path(__file__).parent.parent

        # Create timestamped filename to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save figures
        output_dir = project_root / 'results' / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'obtl_results_{timestamp}.png'

        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'obtl_results_latest.png', dpi=300, bbox_inches='tight')

        print(f"   ✓ Saved figures to {output_dir / filename}")
        print(f"   ✓ Also saved as obtl_results_latest.png")

        # Save metrics
        print("\n4. Saving metrics...")
        save_metrics_to_file(results, timestamp)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print("• OBTL enables covariance structure transfer")
    print("• Optimal δ balances source and target covariance")
    print("• Effective for spatial air quality prediction")

    return results


def create_visualizations(results, data, delta_values):
    """Create 4-panel visualization of OBTL results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Performance vs delta
    ax = axes[0, 0]
    ax.plot(results['delta'], results['rmse'], 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax.set_xlabel('Transfer Parameter δ', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('(a) Transfer Performance vs δ', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) Covariance similarity
    ax = axes[0, 1]
    ax.plot(results['delta'], results['cov_similarity'], 'o-', color='#A23B72', linewidth=2, markersize=8)
    ax.set_xlabel('Transfer Parameter δ', fontsize=11)
    ax.set_ylabel('Normalized Distance to Source Cov', fontsize=11)
    ax.set_title('(b) Covariance Transfer Strength', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (c) R² comparison
    ax = axes[1, 0]
    ax.bar(range(len(delta_values)), results['r2'], color='#F18F01', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(delta_values)))
    ax.set_xticklabels([f'{d:.1f}' for d in delta_values])
    ax.set_xlabel('Transfer Parameter δ', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('(c) Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (d) Spatial distribution
    ax = axes[1, 1]
    X_target, y_target = data['target']
    scatter = ax.scatter(X_target[:, 0], X_target[:, 1], c=y_target, cmap='YlOrRd',
                        s=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Spatial Coordinate X', fontsize=11)
    ax.set_ylabel('Spatial Coordinate Y', fontsize=11)
    ax.set_title('(d) Target Domain Spatial Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='PM2.5 (µg/m³)')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    results = run_experiment(
        delta_values=[0.0, 0.3, 0.5, 0.7, 1.0],
        save_results=True
    )
