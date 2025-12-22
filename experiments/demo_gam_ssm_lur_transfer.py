"""
Demo: Transfer Learning for GAM-SSM-LUR Models

Demonstrates transferring spatial LUR coefficients and temporal SSM dynamics
from a source city (Dublin) to a target city.

Note: This demo requires the gam_ssm_lur package and actual city data.
      It serves as a template for real-world transfer learning experiments.
"""

import numpy as np
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# Add model_transferability to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.gal_ssm_lur import (
        hybrid_transfer,
        TransferableGAMSSM,
        get_transfer_summary,
        GAM_SSM_AVAILABLE
    )

    if not GAM_SSM_AVAILABLE:
        print("⚠️  gam_ssm_lur package not found.")
        print("   Install from: github.com/GabrielOduori/gam_ssm_lur")
        print("   Required dependencies: pygam, gam_ssm_lur")
        print("   This demo shows the transfer learning API structure.")
        GAM_SSM_AVAILABLE = False
    else:
        try:
            from gam_ssm_lur.models.hybrid import HybridGAMSSM
        except ImportError as inner_e:
            print(f"⚠️  gam_ssm_lur imports failed: {inner_e}")
            print("   Install dependencies: pip install pygam")
            print("   Install from: github.com/GabrielOduori/gam_ssm_lur")
            GAM_SSM_AVAILABLE = False

except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("   This demo requires the gam_ssm_lur package.")
    print("   Install from: https://github.com/GabrielOduori/gam_ssm_lur")
    print("   Required dependencies: pip install pygam")
    GAM_SSM_AVAILABLE = False


def save_metrics_to_file(results, timestamp=None):
    """
    Save experiment metrics to JSON, CSV, and LaTeX files.

    Parameters
    ----------
    results : list of dict
        Results list with metrics for each configuration
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
        'experiment': 'gam_ssm_lur_transfer',
        'configurations': [],
        'metrics': {}
    }

    # Add metrics for each configuration
    for result in results:
        config_name = result['name']
        metrics_data['configurations'].append(config_name)
        metrics_data['metrics'][config_name] = {
            'spatial_weight': float(result['spatial_weight']),
            'temporal_weight': float(result['temporal_weight']),
            'rmse': float(result['rmse']),
            'mae': float(result['mae'])
        }

    # Add summary statistics
    best_result = min(results, key=lambda x: x['rmse'])
    baseline_rmse = results[0]['rmse']
    improvement = (baseline_rmse - best_result['rmse']) / baseline_rmse * 100

    metrics_data['summary'] = {
        'best_config': best_result['name'],
        'best_spatial_weight': float(best_result['spatial_weight']),
        'best_temporal_weight': float(best_result['temporal_weight']),
        'best_rmse': float(best_result['rmse']),
        'best_mae': float(best_result['mae']),
        'baseline_rmse': float(baseline_rmse),
        'improvement_percent': float(improvement)
    }

    # 1. Save JSON
    json_filename = f'gam_ssm_lur_metrics_{timestamp}.json'
    json_filepath = metrics_dir / json_filename
    with open(json_filepath, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    latest_json_filepath = metrics_dir / 'gam_ssm_lur_metrics_latest.json'
    with open(latest_json_filepath, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # 2. Save CSV
    csv_filename = f'gam_ssm_lur_metrics_{timestamp}.csv'
    csv_filepath = metrics_dir / csv_filename

    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Spatial_Weight', 'Temporal_Weight', 'RMSE', 'MAE'])
        for result in results:
            writer.writerow([
                result['name'],
                f"{result['spatial_weight']:.2f}",
                f"{result['temporal_weight']:.2f}",
                f"{result['rmse']:.4f}",
                f"{result['mae']:.4f}"
            ])
        # Add summary row
        writer.writerow([])
        writer.writerow(['Summary', '', '', '', ''])
        writer.writerow(['Best Config', best_result['name'], '', '', ''])
        writer.writerow(['Improvement (%)', f"{improvement:.2f}", '', '', ''])

    latest_csv_filepath = metrics_dir / 'gam_ssm_lur_metrics_latest.csv'
    with open(csv_filepath, 'r') as src, open(latest_csv_filepath, 'w') as dst:
        dst.write(src.read())

    # 3. Save LaTeX table
    latex_filename = f'gam_ssm_lur_metrics_{timestamp}.tex'
    latex_filepath = metrics_dir / latex_filename

    with open(latex_filepath, 'w') as f:
        f.write("% GAM-SSM-LUR Transfer Learning Results\n")
        f.write(f"% Generated: {timestamp}\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GAM-SSM-LUR Transfer Learning Performance}\n")
        f.write("\\label{tab:gam_ssm_lur_results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Configuration & $w_s$ & $w_t$ & RMSE & MAE \\\\\n")
        f.write("\\hline\n")

        for result in results:
            # Shorten config name for table
            short_name = result['name'].replace(' transfer', '').replace('No transfer ', '')
            f.write(f"{short_name:20s} & "
                   f"{result['spatial_weight']:.1f} & "
                   f"{result['temporal_weight']:.1f} & "
                   f"{result['rmse']:.4f} & "
                   f"{result['mae']:.4f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\multicolumn{5}{l}{")
        f.write(f"Best: {best_result['name']}, Improvement = {improvement:.1f}\\%")
        f.write("} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    latest_latex_filepath = metrics_dir / 'gam_ssm_lur_metrics_latest.tex'
    with open(latex_filepath, 'r') as src, open(latest_latex_filepath, 'w') as dst:
        dst.write(src.read())

    print(f"   ✓ Saved metrics to:")
    print(f"      - JSON: {json_filepath.name}")
    print(f"      - CSV:  {csv_filepath.name}")
    print(f"      - LaTeX: {latex_filepath.name}")
    print(f"   ✓ Also saved as 'latest' versions")


def create_synthetic_city_data(n_locations=20, n_times=100, n_features=10, seed=42):
    """
    Create synthetic spatio-temporal air quality data.

    Args:
        n_locations: Number of monitoring locations
        n_times: Number of time steps
        n_features: Number of LUR features
        seed: Random seed

    Returns:
        Dict with features, observations, time indices, location indices
    """
    np.random.seed(seed)

    # Generate features for all location-time combinations
    n_obs = n_locations * n_times
    X = np.random.randn(n_obs, n_features)

    # True LUR coefficients
    true_coef = np.random.randn(n_features) * 2

    # Spatial component (LUR)
    spatial = X @ true_coef + 20  # Base level of 20 µg/m³

    # Temporal component (seasonal + trend)
    time_idx = np.repeat(np.arange(n_times), n_locations)
    temporal = 5 * np.sin(2 * np.pi * time_idx / 365.25)  # Seasonal
    temporal = np.tile(temporal[::n_locations], n_locations)  # Broadcast to locations

    # Observations
    y = spatial + temporal + np.random.randn(n_obs) * 2

    # Location indices
    location_idx = np.tile(np.arange(n_locations), n_times)

    return {
        'X': X,
        'y': y,
        'time_index': time_idx,
        'location_index': location_idx,
        'n_locations': n_locations,
        'n_times': n_times
    }


def demonstrate_transfer_learning():
    """Demonstrate GAM-SSM-LUR transfer learning."""

    print("=" * 70)
    print("GAM-SSM-LUR TRANSFER LEARNING - DEMONSTRATION")
    print("=" * 70)
    print("\nTransferring spatial LUR coefficients and temporal SSM dynamics")
    print("from Dublin (source) to Cork (target).\n")

    # Generate synthetic data for two cities
    print("1. Generating synthetic data...")
    dublin_data = create_synthetic_city_data(n_locations=30, n_times=200, seed=42)
    cork_data = create_synthetic_city_data(n_locations=15, n_times=50, seed=123)

    print(f"   Dublin: {dublin_data['n_times']} time steps × {dublin_data['n_locations']} locations")
    print(f"   Cork:   {cork_data['n_times']} time steps × {cork_data['n_locations']} locations")

    # Train source model (Dublin)
    print("\n2. Training source model (Dublin)...")
    dublin_model = TransferableGAMSSM(
        n_splines=10,
        em_max_iter=30,
        scalability_mode='auto'
    )

    dublin_model.fit(
        dublin_data['X'],
        dublin_data['y'],
        dublin_data['time_index'],
        dublin_data['location_index']
    )
    print("   ✓ Dublin model trained")

    # Baseline: Train target from scratch
    print("\n3. Training baseline target model (no transfer)...")
    cork_baseline = TransferableGAMSSM(
        n_splines=10,
        em_max_iter=30,
        scalability_mode='auto'
    )

    cork_baseline.fit(
        cork_data['X'],
        cork_data['y'],
        cork_data['time_index'],
        cork_data['location_index']
    )
    print("   ✓ Cork baseline trained")

    # Transfer learning
    print("\n4. Transfer learning from Dublin to Cork...")
    print("   Testing different transfer weights:\n")

    transfer_configs = [
        {'spatial_weight': 0.0, 'temporal_weight': 0.0, 'name': 'No transfer (baseline)'},
        {'spatial_weight': 0.3, 'temporal_weight': 0.3, 'name': 'Weak transfer'},
        {'spatial_weight': 0.5, 'temporal_weight': 0.5, 'name': 'Balanced transfer'},
        {'spatial_weight': 0.7, 'temporal_weight': 0.7, 'name': 'Strong transfer'},
    ]

    results = []

    for config in transfer_configs:
        cork_transfer = TransferableGAMSSM(
            n_splines=10,
            em_max_iter=30,
            scalability_mode='auto'
        )

        cork_transfer.transfer_from(
            dublin_model,
            cork_data['X'],
            cork_data['y'],
            cork_data['time_index'],
            cork_data['location_index'],
            spatial_weight=config['spatial_weight'],
            temporal_weight=config['temporal_weight']
        )

        # Evaluate
        predictions = cork_transfer.predict(cork_data['X'][:100], return_intervals=False)
        y_true = cork_data['y'][:100]

        if hasattr(predictions, 'total'):
            y_pred = predictions.total.flatten()[:len(y_true)]
        else:
            y_pred = predictions.flatten()[:len(y_true)]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        results.append({
            'name': config['name'],
            'spatial_weight': config['spatial_weight'],
            'temporal_weight': config['temporal_weight'],
            'rmse': rmse,
            'mae': mae
        })

        print(f"   {config['name']:25s} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING SUMMARY")
    print("=" * 70)

    best_result = min(results, key=lambda x: x['rmse'])
    print(f"\nBest configuration: {best_result['name']}")
    print(f"  Spatial weight: {best_result['spatial_weight']}")
    print(f"  Temporal weight: {best_result['temporal_weight']}")
    print(f"  RMSE: {best_result['rmse']:.4f}")
    print(f"  MAE:  {best_result['mae']:.4f}")

    baseline_rmse = results[0]['rmse']
    improvement = (baseline_rmse - best_result['rmse']) / baseline_rmse * 100
    print(f"\nImprovement over baseline: {improvement:.1f}%")

    # Save metrics
    print("\n5. Saving metrics...")
    save_metrics_to_file(results)

    print("\n" + "=" * 70)
    print("Key Components Transferred:")
    print("=" * 70)
    print("✓ LUR Coefficients     - Spatial patterns (land use, traffic, etc.)")
    print("✓ SSM Transition Matrix - Temporal dynamics (seasonal patterns)")
    print("✓ SSM Noise Covariance - Uncertainty structure")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFor real-world transfer:")
    print("1. Load your Dublin GAM-SSM-LUR model")
    print("2. Prepare target city data (LUR features + observations)")
    print("3. Use hybrid_transfer() or TransferableGAMSSM")
    print("4. Fine-tune on target city data")

    return results


if __name__ == "__main__":
    if GAM_SSM_AVAILABLE:
        results = demonstrate_transfer_learning()
    else:
        print("\n" + "=" * 70)
        print("GAM-SSM-LUR Transfer Learning Template")
        print("=" * 70)
        print("\nThis module provides transfer learning for GAM-SSM-LUR models.")
        print("\nKey functions:")
        print("  - transfer_lur_coefficients()  : Transfer spatial LUR parameters")
        print("  - transfer_ssm_dynamics()      : Transfer temporal SSM parameters")
        print("  - hybrid_transfer()            : Complete transfer pipeline")
        print("  - TransferableGAMSSM           : High-level wrapper class")
        print("\nTo run the demo, install gam_ssm_lur:")
        print("  git clone https://github.com/GabrielOduori/gam_ssm_lur.git")
        print("  cd gam_ssm_lur")
        print("  pip install -e .")
