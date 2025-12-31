"""
Training diagnostic visualizations for transfer learning.

This module creates visualizations that show the training/transfer process,
including loss convergence, performance gains/losses, and transfer efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import seaborn as sns


def plot_loss_convergence(
    loss_histories: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    save_pdf: bool = True,
    title: str = "Training Loss Convergence"
) -> Path:
    """
    Plot loss convergence curves for different transfer methods/parameters.

    Shows how training loss decreases over iterations, helping identify:
    - Which methods converge faster
    - Optimal number of training iterations
    - Whether training has plateaued

    Args:
        loss_histories: Dict mapping method names to loss arrays
                       e.g., {'β=0.5': [losses], 'β=1.0': [losses]}
        output_path: Path to save the figure
        save_pdf: If True, also save as PDF
        title: Figure title

    Returns:
        Path to saved PNG file

    Example:
        >>> histories = {
        ...     'β=0.3': np.array([100, 80, 60, 50, 45, 42, 41]),
        ...     'β=0.5': np.array([100, 75, 55, 48, 44, 42, 41]),
        ...     'β=1.0': np.array([100, 70, 50, 45, 43, 42, 41])
        ... }
        >>> plot_loss_convergence(histories, Path('loss_convergence.png'))
    """
    # Set up publication-quality style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Color palette
    colors = sns.color_palette("husl", n_colors=len(loss_histories))

    # Plot each method's loss curve
    for (method, losses), color in zip(loss_histories.items(), colors):
        iterations = np.arange(1, len(losses) + 1)
        ax.plot(iterations, losses, label=method, linewidth=2.5,
                marker='o', markersize=4, alpha=0.8, color=color)

    # Styling
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Negative Log Marginal Likelihood', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Log scale if losses vary by orders of magnitude
    loss_range = max(max(losses) for losses in loss_histories.values()) / \
                 min(min(losses) for losses in loss_histories.values())
    if loss_range > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Log(Negative Log Marginal Likelihood)', fontweight='bold')

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = Path('loss_convergence.png')
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")

    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   ✓ Saved: {pdf_path.name}")

    plt.close()
    return output_path


def plot_transfer_gain_loss(
    results: Dict[str, List[Dict]],
    baseline_metric: float,
    output_path: Optional[Path] = None,
    save_pdf: bool = True,
    metric: str = 'rmse'
) -> Path:
    """
    Visualize transfer learning gains/losses relative to baseline.

    Creates a waterfall chart showing how each transfer parameter
    improves (gain) or degrades (loss) performance compared to baseline.

    Args:
        results: Dict with transfer results for each parameter value
                 e.g., {'results': [{'lambda': 0.3, 'rmse': 10.2}, ...]}
        baseline_metric: Baseline metric value (e.g., no transfer RMSE)
        output_path: Path to save the figure
        save_pdf: If True, also save as PDF
        metric: Metric to analyze ('rmse', 'mae', or 'r2')

    Returns:
        Path to saved PNG file

    Example:
        >>> results = {
        ...     'results': [
        ...         {'lambda': 0.0, 'rmse': 15.0},
        ...         {'lambda': 0.5, 'rmse': 12.0},
        ...         {'lambda': 1.0, 'rmse': 10.0}
        ...     ]
        ... }
        >>> plot_transfer_gain_loss(results, baseline_metric=15.0)
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Extract data
    result_list = results['results']
    param_name = 'lambda' if 'lambda' in result_list[0] else 'delta'

    params = [r[param_name] for r in result_list]
    metrics = [r[metric] for r in result_list]

    # Calculate improvement percentages
    if metric in ['rmse', 'mae']:
        # Lower is better
        improvements = [(baseline_metric - m) / baseline_metric * 100 for m in metrics]
    else:  # r2
        # Higher is better
        improvements = [(m - baseline_metric) / abs(baseline_metric) * 100 for m in metrics]

    # Colors: green for gain, red for loss
    colors = ['#029E73' if imp > 0 else '#F44336' for imp in improvements]

    # Create bar chart
    bars = ax.bar(range(len(params)), improvements, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, imp, metric_val) in enumerate(zip(bars, improvements, metrics)):
        height = bar.get_height()
        label_y = height + (2 if height > 0 else -5)

        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{imp:+.1f}%\n({metric.upper()}={metric_val:.2f})',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel(f'Transfer Parameter ({param_name})', fontweight='bold', fontsize=14)
    ax.set_ylabel('Performance Change (%)', fontweight='bold', fontsize=14)
    ax.set_title(f'Transfer Learning Gain/Loss Analysis\n(Baseline {metric.upper()} = {baseline_metric:.2f})',
                 fontweight='bold', pad=20)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([f'{param_name}={p:.1f}' for p in params])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#029E73', alpha=0.7, edgecolor='black', label='Positive Transfer (Gain)'),
        Patch(facecolor='#F44336', alpha=0.7, edgecolor='black', label='Negative Transfer (Loss)')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, shadow=True)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = Path('transfer_gain_loss.png')
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")

    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   ✓ Saved: {pdf_path.name}")

    plt.close()
    return output_path


def plot_transfer_weight_evolution(
    obtl_results: List[Dict],
    output_path: Optional[Path] = None,
    save_pdf: bool = True
) -> Path:
    """
    Visualize how source/target weights change with delta parameter.

    Shows the Bayesian weighting mechanism in OBTL - as delta increases,
    source domain gets more weight in the transferred model.

    Args:
        obtl_results: List of OBTL results with weight_source/weight_target
        output_path: Path to save the figure
        save_pdf: If True, also save as PDF

    Returns:
        Path to saved PNG file

    Example:
        >>> results = [
        ...     {'delta': 0.3, 'weight_source': 0.2, 'weight_target': 0.8},
        ...     {'delta': 1.0, 'weight_source': 0.5, 'weight_target': 0.5}
        ... ]
        >>> plot_transfer_weight_evolution(results)
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'font.family': 'serif'
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Extract data
    deltas = [r['delta'] for r in obtl_results]
    weights_source = [r['weight_source'] for r in obtl_results]
    weights_target = [r['weight_target'] for r in obtl_results]
    rmse_values = [r['rmse'] for r in obtl_results]

    # Left panel: Stacked area chart
    ax1.fill_between(deltas, 0, weights_source, alpha=0.6, color='#0173B2',
                     label='Source Domain Weight')
    ax1.fill_between(deltas, weights_source,
                     np.array(weights_source) + np.array(weights_target),
                     alpha=0.6, color='#DE8F05', label='Target Domain Weight')

    ax1.set_xlabel('Transfer Strength (δ)', fontweight='bold')
    ax1.set_ylabel('Weight Proportion', fontweight='bold')
    ax1.set_title('Bayesian Weight Allocation', fontweight='bold', pad=10)
    ax1.legend(loc='center left', frameon=True, shadow=True)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right panel: Weight ratio vs performance
    weight_ratios = np.array(weights_source) / np.array(weights_target)

    scatter = ax2.scatter(weight_ratios, rmse_values, s=200, c=deltas,
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)

    # Add delta labels
    for delta, ratio, rmse in zip(deltas, weight_ratios, rmse_values):
        ax2.annotate(f'δ={delta:.1f}', (ratio, rmse),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax2.set_xlabel('Weight Ratio (Source/Target)', fontweight='bold')
    ax2.set_ylabel('RMSE (µg/m³)', fontweight='bold')
    ax2.set_title('Weight Balance vs Performance', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('δ parameter', fontweight='bold')

    # Main title
    fig.suptitle('OBTL Transfer Weight Evolution', fontsize=18, fontweight='bold', y=1.00)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = Path('transfer_weight_evolution.png')
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")

    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   ✓ Saved: {pdf_path.name}")

    plt.close()
    return output_path


if __name__ == '__main__':
    # Test the functions with example data
    print("Testing training diagnostic visualizations...")

    # 1. Loss convergence
    print("\n1. Loss convergence curves...")
    loss_histories = {
        'β=0.3': 100 * np.exp(-0.05 * np.arange(100)) + np.random.normal(0, 2, 100),
        'β=0.5': 100 * np.exp(-0.06 * np.arange(100)) + np.random.normal(0, 2, 100),
        'β=1.0': 100 * np.exp(-0.07 * np.arange(100)) + np.random.normal(0, 2, 100),
    }
    plot_loss_convergence(loss_histories)

    # 2. Transfer gain/loss
    print("\n2. Transfer gain/loss analysis...")
    results = {
        'results': [
            {'lambda': 0.0, 'rmse': 15.5, 'mae': 12.0, 'r2': 0.25},
            {'lambda': 0.3, 'rmse': 13.2, 'mae': 10.5, 'r2': 0.45},
            {'lambda': 0.5, 'rmse': 11.8, 'mae': 9.2, 'r2': 0.58},
            {'lambda': 0.7, 'rmse': 12.5, 'mae': 9.8, 'r2': 0.52},
            {'lambda': 1.0, 'rmse': 14.0, 'mae': 11.0, 'r2': 0.35},
        ]
    }
    plot_transfer_gain_loss(results, baseline_metric=15.5, metric='rmse')

    # 3. Transfer weight evolution
    print("\n3. Transfer weight evolution...")
    obtl_results = [
        {'delta': 0.3, 'weight_source': 0.23, 'weight_target': 0.77, 'rmse': 12.5},
        {'delta': 0.5, 'weight_source': 0.33, 'weight_target': 0.67, 'rmse': 11.2},
        {'delta': 0.7, 'weight_source': 0.41, 'weight_target': 0.59, 'rmse': 10.8},
        {'delta': 1.0, 'weight_source': 0.50, 'weight_target': 0.50, 'rmse': 11.5},
    ]
    plot_transfer_weight_evolution(obtl_results)

    print("\nAll test visualizations generated!")
