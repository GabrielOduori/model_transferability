"""
Model Transfer Learning Experiments
====================================

Transfer pre-trained Source domain models (FusionGP and GAM-SSM-LUR) to
synthetic Target domain data using Prior Tempering and OBTL paradigms.

This refactored version uses modular components for clean, maintainable code.
"""

import sys
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import new modular components
from config import ExperimentConfig
from data import SyntheticDataGenerator, ModelLoader
from transfer import PriorTemperingExperiment, OBTLExperiment
from reporting import ResultsExporter, VisualizationManager


def main():
    """Run transfer learning experiments using modular components."""

    print("="*70)
    print("MODEL TRANSFER LEARNING EXPERIMENTS")
    print("="*70)
    print("\nTransfer Scenario:")
    print("  Source: Pre-trained models (real Source domain data)")
    print("    - FusionGP")
    print("    - GAM-SSM-LUR")
    print("  Target: Synthetic Target domain data")
    print("  Methods: Prior Tempering, OBTL")
    print("="*70)

    # ========== Configuration ==========
    config = ExperimentConfig()
    print(f"\n Experiment Configuration:")
    print(f"   Target samples: {config.n_target}")
    print(f"   Test samples: {config.n_test}")
    print(f"   Lambda values: {config.lambda_values}")
    print(f"   Delta values: {config.delta_values}")
    print(f"   Random seed: {config.seed}")

    # ========== Generate Synthetic Data ==========
    print(f"\n Generating synthetic Target domain data...")
    data_gen = SyntheticDataGenerator(seed=config.seed)
    target_data_dict = data_gen.generate(
        n_target=config.n_target,
        n_test=config.n_test,
        force_regenerate=False
    )

    target_data = target_data_dict['target']
    test_data = target_data_dict['test']

    print(f"   ✓ Target: {target_data['X'].shape[0]} samples")
    print(f"   ✓ Test:   {test_data['X'].shape[0]} samples")

    # ========== Standardize Target Data ==========
    # Normalize to zero mean, unit variance to improve transfer learning
    # This addresses the scale mismatch between source and target domains
    print(f"\n Standardizing target data...")
    print(f"   Before: y_target mean={target_data['y'].mean():.2f}, std={target_data['y'].std():.2f}")
    print(f"   Before: y_test mean={test_data['y'].mean():.2f}, std={test_data['y'].std():.2f}")

    # Compute statistics on training data only
    y_target_mean = target_data['y'].mean()
    y_target_std = target_data['y'].std()

    # Apply standardization
    target_data['y'] = (target_data['y'] - y_target_mean) / y_target_std
    test_data['y'] = (test_data['y'] - y_target_mean) / y_target_std

    print(f"   After: y_target mean={target_data['y'].mean():.2f}, std={target_data['y'].std():.2f}")
    print(f"   After: y_test mean={test_data['y'].mean():.2f}, std={test_data['y'].std():.2f}")
    print(f"   ✓ Data standardized (mean=0, std=1)")
    print(f"   ✓ Normalization: y_norm = (y - {y_target_mean:.2f}) / {y_target_std:.2f}")

    # ========== Load Source Models ==========
    loader = ModelLoader()

    # Verify model files exist
    status = loader.verify_model_files(config)
    if not all(status.values()):
        print("\n ERROR: Some model files are missing!")
        return

    # Load FusionGP
    try:
        print(f"\n Loading FusionGP model...")
        fusiongp_model, fusiongp_likelihood = loader.load_fusiongp(config.fusiongp_path)
        print(f"   ✓ FusionGP loaded successfully")
    except Exception as e:
        print(f"\n Error loading FusionGP: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load GAM-SSM-LUR
    try:
        print(f"\n Loading GAM-SSM-LUR model...")
        gam_model, gam_data = loader.load_gam_ssm_lur(
            config.gam_path,
            config.ssm_path,
            config.gam_data_path
        )
        print(f"   ✓ GAM-SSM-LUR loaded successfully")
    except Exception as e:
        print(f"\n Error loading GAM-SSM-LUR: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== Run Transfer Experiments ==========

    # Experiment 1: FusionGP + Prior Tempering
    fusiongp_pt_exp = PriorTemperingExperiment(
        source_model=fusiongp_model,
        source_likelihood=fusiongp_likelihood,
        model_type='fusiongp'
    )
    results_fusiongp_pt = fusiongp_pt_exp.run(
        target_data=target_data,
        test_data=test_data,
        param_values=config.lambda_values
    )

    # Experiment 2: GAM-SSM-LUR + Prior Tempering
    gam_pt_exp = PriorTemperingExperiment(
        source_model=gam_model,
        source_data=gam_data,
        model_type='gam_ssm_lur'
    )
    results_gam_pt = gam_pt_exp.run(
        target_data=target_data,
        test_data=test_data,
        param_values=config.lambda_values
    )

    # Experiment 3: FusionGP + OBTL
    fusiongp_obtl_exp = OBTLExperiment(
        source_model=fusiongp_model,
        source_likelihood=fusiongp_likelihood,
        model_type='fusiongp',
        n_inducing_points=config.n_inducing_points,
        nu_0=config.nu_0
    )
    results_fusiongp_obtl = fusiongp_obtl_exp.run(
        target_data=target_data,
        test_data=test_data,
        param_values=config.delta_values
    )

    # Experiment 4: GAM-SSM-LUR + OBTL
    gam_obtl_exp = OBTLExperiment(
        source_model=gam_model,
        source_data=gam_data,
        model_type='gam_ssm_lur',
        n_inducing_points=config.n_inducing_points,
        nu_0=config.nu_0
    )
    results_gam_obtl = gam_obtl_exp.run(
        target_data=target_data,
        test_data=test_data,
        param_values=config.delta_values
    )

    # ========== Package Results ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).parent.parent / 'results'
    experiment_dir = base_results_dir / f'experiment_{timestamp}'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': timestamp,
        'source': 'Real Source Domain Models',
        'target': 'Synthetic Target Domain',
        'config': config.to_dict(),
        'fusiongp_prior_tempering': results_fusiongp_pt,
        'gam_ssm_lur_prior_tempering': results_gam_pt,
        'fusiongp_obtl': results_fusiongp_obtl,
        'gam_ssm_lur_obtl': results_gam_obtl
    }

    # ========== Export Results ==========
    exporter = ResultsExporter(experiment_dir, timestamp)
    exporter.export_all(all_results)

    # ========== Create Visualizations ==========
    viz_manager = VisualizationManager(experiment_dir, timestamp)
    viz_manager.create_all(all_results, test_data=test_data)

    # ========== Print Summary ==========
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n Experiment folder: {experiment_dir.relative_to(base_results_dir.parent)}")

    print(f"\n Best Results Summary:")
    print(f"\n  FusionGP:")
    print(f"    Prior Tempering: λ={results_fusiongp_pt['best']['lambda']}, "
          f"RMSE={results_fusiongp_pt['best']['rmse']:.2f} µg/m³, "
          f"R²={results_fusiongp_pt['best']['r2']:.4f}")
    print(f"    OBTL:            δ={results_fusiongp_obtl['best']['delta']}, "
          f"RMSE={results_fusiongp_obtl['best']['rmse']:.2f} µg/m³, "
          f"R²={results_fusiongp_obtl['best']['r2']:.4f}")

    print(f"\n  GAM-SSM-LUR:")
    print(f"    Prior Tempering: λ={results_gam_pt['best']['lambda']}, "
          f"RMSE={results_gam_pt['best']['rmse']:.2f} µg/m³, "
          f"R²={results_gam_pt['best']['r2']:.4f}")
    print(f"    OBTL:            δ={results_gam_obtl['best']['delta']}, "
          f"RMSE={results_gam_obtl['best']['rmse']:.2f} µg/m³, "
          f"R²={results_gam_obtl['best']['r2']:.4f}")

    # Print directory structure
    print(f"\n{'='*70}")
    print("EXPERIMENT STRUCTURE:")
    print(f"{'='*70}")
    print(f"\n{experiment_dir.name}/")
    print(f"├── results_{timestamp}.json              # All experimental results (JSON)")
    print(f"├── tables/                               # Results tables (CSV + LaTeX)")
    print(f"│   ├── prior_tempering_results.csv")
    print(f"│   ├── prior_tempering_results.tex")
    print(f"│   ├── obtl_results.csv")
    print(f"│   ├── obtl_results.tex")
    print(f"│   ├── summary_best_results.csv")
    print(f"│   └── summary_best_results.tex")
    print(f"└── figures/                              # Publication-quality plots (13 files)")
    print(f"    ├── prior_tempering_fancy_{timestamp}.png")
    print(f"    ├── obtl_fancy_{timestamp}.png")
    print(f"    ├── summary_comparison_{timestamp}.png")
    print(f"    ├── prior_tempering_concept_{timestamp}.{{png,pdf}}")
    print(f"    ├── spatial_comparison_{timestamp}.{{png,pdf}}")
    print(f"    ├── pt_gain_loss_{timestamp}.{{png,pdf}}")
    print(f"    ├── obtl_gain_loss_{timestamp}.{{png,pdf}}")
    print(f"    └── obtl_weights_{timestamp}.{{png,pdf}}")

    print(f"\n{'='*70}")
    print(f" Success! Results saved to: {experiment_dir.relative_to(base_results_dir.parent)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
