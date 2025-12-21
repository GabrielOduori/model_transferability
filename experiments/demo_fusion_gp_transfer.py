"""
Demo: Transfer Learning for FusionSVGP Models

Demonstrates transferring multi-source GP from Dublin to target city.
Transfers kernel hyperparameters, inducing points, and likelihood parameters.

Note: This demo requires the fusiongp package and actual city data.
      It serves as a template for real-world transfer learning experiments.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add model_transferability to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.fusion_gp_transfer import (
        hybrid_fusion_transfer,
        TransferableFusionSVGP,
        get_transfer_summary,
        FUSION_GP_AVAILABLE
    )

    if not FUSION_GP_AVAILABLE:
        print("⚠️  fusiongp package not found.")
        print("   Install from: github.com/GabrielOduori/fusionGP2")
        print("   This demo shows the transfer learning API structure.")
        sys.exit(0)

    from models.svgp import FusionSVGP

except ImportError as e:
    print(f"Import error: {e}")
    print("\nThis demo requires the fusiongp package.")
    print("Install from: https://github.com/GabrielOduori/fusionGP2")
    sys.exit(1)


def create_synthetic_multisource_data(n_points=100, n_sources=3, seed=42):
    """
    Create synthetic multi-source spatio-temporal data.

    Args:
        n_points: Number of observation locations
        n_sources: Number of data sources
        seed: Random seed

    Returns:
        Dict with x (locations), y (observations), source_masks
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Spatio-temporal locations: [lat, lon, time] normalized to [0, 1]
    x = torch.rand(n_points, 3)

    # Simulate true NO2 field with spatial + temporal patterns
    # Spatial component
    spatial = 30 + 20 * torch.sin(2 * np.pi * x[:, 0]) * torch.cos(2 * np.pi * x[:, 1])

    # Temporal component
    temporal = 10 * torch.sin(2 * np.pi * x[:, 2])

    # True latent NO2
    true_no2 = spatial + temporal

    # Multi-source observations with different noise levels
    y = torch.zeros(n_points, n_sources)
    source_masks = torch.zeros(n_points, n_sources, dtype=torch.bool)

    # Source 0: EPA (low noise, full coverage)
    y[:, 0] = true_no2 + torch.randn(n_points) * 2.0
    source_masks[:, 0] = True

    # Source 1: Low-cost (higher noise, calibration bias, partial coverage)
    n_lc = n_points // 2
    lc_indices = torch.randperm(n_points)[:n_lc]
    # Calibration: y = 1.2 * f + 5
    y[lc_indices, 1] = 1.2 * true_no2[lc_indices] + 5 + torch.randn(n_lc) * 5.0
    source_masks[lc_indices, 1] = True

    # Source 2: Satellite (moderate noise, sparse coverage)
    n_sat = n_points // 3
    sat_indices = torch.randperm(n_points)[:n_sat]
    y[sat_indices, 2] = true_no2[sat_indices] + torch.randn(n_sat) * 3.0
    source_masks[sat_indices, 2] = True

    return {
        'x': x,
        'y': y,
        'source_masks': source_masks,
        'true_no2': true_no2
    }


def demonstrate_transfer_learning():
    """Demonstrate FusionSVGP transfer learning."""

    print("=" * 70)
    print("FUSIONSVGP TRANSFER LEARNING - DEMONSTRATION")
    print("=" * 70)
    print("\nTransferring multi-source GP from Dublin (source) to Cork (target).")
    print("Transfers kernel hyperparameters, inducing points, and likelihood params.\n")

    # Generate synthetic data for two cities
    print("1. Generating synthetic multi-source data...")
    dublin_data = create_synthetic_multisource_data(n_points=200, seed=42)
    cork_data = create_synthetic_multisource_data(n_points=100, seed=123)

    print(f"   Dublin: {len(dublin_data['x'])} observations")
    print(f"   Cork:   {len(cork_data['x'])} observations")

    # Train source model (Dublin)
    print("\n2. Training source model (Dublin)...")
    dublin_model = TransferableFusionSVGP(
        n_inducing=50,
        kernel_type='matern32',
        sources=['epa', 'low_cost', 'satellite']
    )

    dublin_model.initialize_inducing_points(dublin_data['x'], method='kmeans')
    losses = dublin_model.train(
        dublin_data['x'],
        dublin_data['y'],
        dublin_data['source_masks'],
        epochs=50,
        lr=0.01
    )
    print(f"   ✓ Dublin model trained (final loss: {losses[-1]:.2f})")

    # Baseline: Train target from scratch
    print("\n3. Training baseline target model (no transfer)...")
    cork_baseline = TransferableFusionSVGP(
        n_inducing=50,
        kernel_type='matern32',
        sources=['epa', 'low_cost', 'satellite']
    )

    cork_baseline.initialize_inducing_points(cork_data['x'], method='kmeans')
    baseline_losses = cork_baseline.train(
        cork_data['x'],
        cork_data['y'],
        cork_data['source_masks'],
        epochs=50,
        lr=0.01
    )
    print(f"   ✓ Cork baseline trained (final loss: {baseline_losses[-1]:.2f})")

    # Transfer learning
    print("\n4. Transfer learning from Dublin to Cork...")
    print("   Testing different transfer weights:\n")

    transfer_configs = [
        {'kernel': 0.0, 'inducing': 0.0, 'likelihood': 0.0, 'name': 'No transfer (baseline)'},
        {'kernel': 0.3, 'inducing': 0.2, 'likelihood': 0.3, 'name': 'Weak transfer'},
        {'kernel': 0.5, 'inducing': 0.3, 'likelihood': 0.5, 'name': 'Balanced transfer'},
        {'kernel': 0.7, 'inducing': 0.5, 'likelihood': 0.7, 'name': 'Strong transfer'},
    ]

    results = []

    for config in transfer_configs:
        cork_transfer = TransferableFusionSVGP(
            n_inducing=50,
            kernel_type='matern32',
            sources=['epa', 'low_cost', 'satellite']
        )

        cork_transfer.transfer_from(
            dublin_model,
            cork_data['x'],
            cork_data['y'],
            cork_data['source_masks'],
            kernel_weight=config['kernel'],
            inducing_weight=config['inducing'],
            likelihood_weight=config['likelihood']
        )

        # Evaluate on test subset
        test_indices = torch.randperm(len(cork_data['x']))[:30]
        test_x = cork_data['x'][test_indices]
        test_true = cork_data['true_no2'][test_indices]

        with torch.no_grad():
            pred_mean, pred_var = cork_transfer.predict(test_x, include_noise=False)

        rmse = torch.sqrt(torch.mean((pred_mean - test_true) ** 2)).item()
        mae = torch.mean(torch.abs(pred_mean - test_true)).item()

        results.append({
            'name': config['name'],
            'kernel_weight': config['kernel'],
            'inducing_weight': config['inducing'],
            'likelihood_weight': config['likelihood'],
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
    print(f"  Kernel weight: {best_result['kernel_weight']}")
    print(f"  Inducing weight: {best_result['inducing_weight']}")
    print(f"  Likelihood weight: {best_result['likelihood_weight']}")
    print(f"  RMSE: {best_result['rmse']:.4f}")
    print(f"  MAE:  {best_result['mae']:.4f}")

    baseline_rmse = results[0]['rmse']
    improvement = (baseline_rmse - best_result['rmse']) / baseline_rmse * 100
    print(f"\nImprovement over baseline: {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("Key Components Transferred:")
    print("=" * 70)
    print("✓ Kernel Hyperparameters - Spatial/temporal lengthscales, outputscale")
    print("✓ Inducing Points        - Optimal locations for variational approximation")
    print("✓ Likelihood Parameters  - Source-specific noise, low-cost calibration")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFor real-world transfer:")
    print("1. Load your Dublin FusionSVGP model")
    print("2. Prepare target city multi-source data")
    print("3. Use hybrid_fusion_transfer() or TransferableFusionSVGP")
    print("4. Fine-tune on target city data")

    return results


if __name__ == "__main__":
    if FUSION_GP_AVAILABLE:
        results = demonstrate_transfer_learning()
    else:
        print("\n" + "=" * 70)
        print("FusionSVGP Transfer Learning Template")
        print("=" * 70)
        print("\nThis module provides transfer learning for FusionSVGP models.")
        print("\nKey functions:")
        print("  - transfer_kernel_hyperparameters() : Transfer spatial/temporal lengthscales")
        print("  - transfer_inducing_points()        : Transfer inducing point locations")
        print("  - transfer_likelihood_parameters()  : Transfer noise and calibration")
        print("  - hybrid_fusion_transfer()          : Complete transfer pipeline")
        print("  - TransferableFusionSVGP            : High-level wrapper class")
        print("\nTo run the demo, install fusiongp:")
        print("  git clone https://github.com/GabrielOduori/fusionGP2.git")
        print("  cd fusionGP2/fusiongp")
        print("  pip install -e .")
