#!/usr/bin/env python3
"""
Diagnostic script to investigate transfer learning issues:
1. Check data scale mismatch between source and synthetic target
2. Examine prediction distributions
3. Identify why R² is so negative
"""

import numpy as np
import torch
from pathlib import Path

# Load synthetic target data
target_data_path = Path("data/synthetic_target/target_data_seed42.npz")
target_data = np.load(target_data_path)

print("="*70)
print("SYNTHETIC TARGET DATA STATISTICS")
print("="*70)
print(f"X_target shape: {target_data['X_target'].shape}")
print(f"y_target shape: {target_data['y_target'].shape}")
print(f"X_test shape: {target_data['X_test'].shape}")
print(f"y_test shape: {target_data['y_test'].shape}")
print()
print(f"Target NO₂ statistics:")
print(f"  Mean:   {target_data['y_target'].mean():.2f} µg/m³")
print(f"  Std:    {target_data['y_target'].std():.2f} µg/m³")
print(f"  Min:    {target_data['y_target'].min():.2f} µg/m³")
print(f"  Max:    {target_data['y_target'].max():.2f} µg/m³")
print()
print(f"Test NO₂ statistics:")
print(f"  Mean:   {target_data['y_test'].mean():.2f} µg/m³")
print(f"  Std:    {target_data['y_test'].std():.2f} µg/m³")
print(f"  Min:    {target_data['y_test'].min():.2f} µg/m³")
print(f"  Max:    {target_data['y_test'].max():.2f} µg/m³")

# Load source model training data
print("\n" + "="*70)
print("SOURCE MODEL TRAINING DATA STATISTICS")
print("="*70)

# FusionGP source data (from inducing points pseudo-data)
fusiongp_path = Path("models/fusiongp/dublin/fusiongp_model.pth")
if fusiongp_path.exists():
    checkpoint = torch.load(fusiongp_path, weights_only=False)
    print(f"\nFusionGP source model:")
    print(f"  Inducing points: {checkpoint['model_state_dict']['variational_strategy.inducing_points'].shape}")
    # Can't directly get training data scale from checkpoint

# GAM-SSM-LUR source data
gam_data_path = Path("models/gam_ssm_lur/dublin/training_data.npz")
if gam_data_path.exists():
    gam_data = np.load(gam_data_path)
    print(f"\nGAM-SSM-LUR source data:")
    print(f"  X_train shape: {gam_data['X_train'].shape}")
    print(f"  y_train shape: {gam_data['y_train'].shape}")
    print(f"  Source NO₂ statistics:")
    print(f"    Mean:   {gam_data['y_train'].mean():.2f} µg/m³")
    print(f"    Std:    {gam_data['y_train'].std():.2f} µg/m³")
    print(f"    Min:    {gam_data['y_train'].min():.2f} µg/m³")
    print(f"    Max:    {gam_data['y_train'].max():.2f} µg/m³")

# Load latest experimental results
print("\n" + "="*70)
print("LATEST EXPERIMENTAL RESULTS")
print("="*70)

import json
results_path = Path("results/experiment_20251229_224352/results_20251229_224352.json")
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    print("\nPrior Tempering Results:")
    print(f"  FusionGP best (λ={results['fusiongp_prior_tempering']['best']['lambda']}):")
    print(f"    RMSE: {results['fusiongp_prior_tempering']['best']['rmse']:.2f} µg/m³")
    print(f"    R²:   {results['fusiongp_prior_tempering']['best']['r2']:.4f}")

    print(f"\n  GAM-SSM-LUR best (λ={results['gam_ssm_lur_prior_tempering']['best']['lambda']}):")
    print(f"    RMSE: {results['gam_ssm_lur_prior_tempering']['best']['rmse']:.2f} µg/m³")
    print(f"    R²:   {results['gam_ssm_lur_prior_tempering']['best']['r2']:.4f}")

    print("\nOBTL Results:")
    print(f"  FusionGP best (δ={results['fusiongp_obtl']['best']['delta']}):")
    print(f"    RMSE: {results['fusiongp_obtl']['best']['rmse']:.2f} µg/m³")
    print(f"    R²:   {results['fusiongp_obtl']['best']['r2']:.4f}")

    print(f"\n  GAM-SSM-LUR best (δ={results['gam_ssm_lur_obtl']['best']['delta']}):")
    print(f"    RMSE: {results['gam_ssm_lur_obtl']['best']['rmse']:.2f} µg/m³")
    print(f"    R²:   {results['gam_ssm_lur_obtl']['best']['r2']:.4f}")

# Compute what R² means
print("\n" + "="*70)
print("R² INTERPRETATION")
print("="*70)
print(f"Test data variance: {target_data['y_test'].var():.2f}")
print(f"Test data mean: {target_data['y_test'].mean():.2f}")
print(f"\nMean baseline RMSE would be: {target_data['y_test'].std():.2f}")
print(f"\nCurrent best RMSE (Prior Tempering): 4.39 µg/m³")
print(f"Current worst RMSE (OBTL): 14.03 µg/m³")
print(f"\nFor R² to be positive, RMSE must be < {target_data['y_test'].std():.2f} µg/m³")
print(f"\nR² = 1 - (RMSE² / variance)")
print(f"R² with RMSE=4.39: {1 - (4.39**2 / target_data['y_test'].var()):.4f}")
print(f"R² with RMSE=14.03: {1 - (14.03**2 / target_data['y_test'].var()):.4f}")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print("✓ R² clipping removed - will see true values in next run")
print("• Check if source and target data scales match")
print("• OBTL performs 3x worse than Prior Tempering (RMSE 14 vs 5)")
print("• GAM-SSM-LUR Prior Tempering not varying with λ (still needs debugging)")
