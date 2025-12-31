# Air Quality Model Transferability

This repository implements transfer learning methods for air quality models across different geographic regions, supporting the research presented in the thesis chapter on probabilistic transfer learning.

## Research Questions

**RQ1**: Can probabilistic transfer learning methods enable air quality models to generalize across different geographic regions with limited target domain data?

**RQ2**: How do different transfer learning approaches compare in terms of prediction accuracy, uncertainty calibration, and computational efficiency for air quality applications?

## Overview

This project implements three probabilistic transfer learning paradigms:

1. **Bayesian Prior Transfer with Tempering** ✅ - Adaptive knowledge transfer using temperature parameter β
2. **Optimal Bayesian Transfer Learning (OBTL)** ✅ - Joint Wishart priors for covariance structure transfer
3. **Deep Probabilistic Transfer Regression (DPTR)** ✅ - VAE-based domain adaptation for missing data scenarios

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Prior Tempering | ✅ Complete | Hyperparameter transfer with β tempering |
| OBTL | ✅ Complete | Covariance structure transfer (theoretical framework) |
| DPTR | ✅ Complete | VAE-based feature alignment for sensor adaptation |
| GAM-SSM-LUR Transfer | ✅ Complete | LUR coefficient + SSM dynamics transfer |
| FusionSVGP Transfer | ✅ Complete | Multi-source GP kernel + inducing point transfer |
| GP Models | ✅ Complete | Baseline and spatial-temporal GPs |
| Evaluation Metrics | ✅ Complete | KL divergence, PICP, RMSE, calibration |
| Demo Experiments | ✅ Complete | All transfer methods with synthetic data |
| Real Model Transfer | ✅ Complete | 2×2 framework (FusionGP + GAM-SSM-LUR × Prior Tempering + OBTL) |

## Models

- **GPyTorch Gaussian Process**: Spatial-temporal air quality prediction with uncertainty quantification
- **GAM-SSM-LUR**: Hybrid Generalized Additive Model–State Space Model with Land Use Regression (requires [gam_ssm_lur](https://github.com/GabrielOduori/gam_ssm_lur))
- **FusionSVGP**: Multi-source Sparse Variational GP for sensor fusion (requires [fusiongp](https://github.com/GabrielOduori/fusiongp))

## Project Structure

```
model_transferability/
├── src/
│   ├── transfer_methods/     # Transfer learning implementations
│   │   ├── prior_tempering.py
│   │   ├── obtl.py
│   │   └── dptr.py
│   ├── models/               # Base model implementations
│   │   ├── gp_model.py
│   │   └── gal_ssm_lur.py
│   ├── evaluation/           # Metrics and evaluation
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── data/                 # Data loading and preprocessing
│   │   └── loaders.py
│   └── utils/                # Utility functions
│       └── helpers.py
├── experiments/              # Experimental scripts
│   ├── rq1_cross_regional.py
│   └── rq2_sensor_adaptation.py
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── results/                  # Experimental results
│   ├── figures/
│   └── metrics/
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/GabrielOduori/model_transferability.git
cd model_transferability

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Demo Experiments

```bash
# Prior Tempering demo (RQ1: Hyperparameter transfer)
python -m experiments.demo_prior_tempering

# OBTL demo (RQ1: Covariance structure transfer)
python -m experiments.demo_obtl

# DPTR demo (RQ2: Sensor adaptation with feature mismatch)
python -m experiments.demo_dptr

# GAM-SSM-LUR transfer (requires gam_ssm_lur package)
python -m experiments.demo_gam_ssm_lur_transfer

# FusionSVGP transfer (requires fusiongp package)
python -m experiments.demo_fusion_gp_transfer
```

Results are saved to `results/figures/`.

### Reproduce Thesis Results

#### Option 1: Use Saved Synthetic Data (Recommended)

```bash
# Demo: Run transfer learning with saved synthetic data
python experiments/run_transfer_with_saved_data.py

# This script demonstrates:
# - Loading saved synthetic data from data/synthetic_target/target_data_seed42.npz
# - Training baseline GP (no transfer, λ=0.0)
# - Simulating Prior Tempering transfer (λ=0.3, 0.5, 0.7, 1.0)
# - Computing RMSE, MAE, R² metrics

# Verify saved data reproducibility
python scripts/verify_synthetic_data.py
```

#### Option 2: Full Experiment with Real Source Models

```bash
# Core experiment producing ALL thesis results
python experiments/run_model_transfer_experiments.py

# This script:
# 1. Loads pre-trained source models (FusionGP, GAM-SSM-LUR)
# 2. Generates/loads synthetic target data (seed=42)
# 3. Runs Prior Tempering and OBTL transfer
# 4. Creates publication-quality visualizations (seaborn)
# 5. Saves results and figures

# Outputs (organized in single experiment folder):
# results/experiment_TIMESTAMP/
#   ├── results.json                          # All results (JSON)
#   ├── tables/                               # Results tables (CSV + LaTeX)
#   │   ├── prior_tempering_results.csv
#   │   ├── prior_tempering_results.tex
#   │   ├── obtl_results.csv
#   │   ├── obtl_results.tex
#   │   ├── summary_best_results.csv
#   │   └── summary_best_results.tex
#   └── figures/                              # Publication-quality plots
#       ├── prior_tempering_fancy_*.png
#       ├── obtl_fancy_*.png
#       └── summary_heatmap_*.png
# data/synthetic_target/target_data_seed42.npz (saved for reproducibility)
```

**Data Availability**:
- **Synthetic Target Data**: [`data/synthetic_target/target_data_seed42.npz`](data/synthetic_target/target_data_seed42.npz) (4.8 KB, seed=42, 50 train + 100 test samples)
- **Results JSON**: [`results/real_model_transfer/real_transfer_20251225_193157.json`](results/real_model_transfer/real_transfer_20251225_193157.json)
- **Source Models**: Not publicly available (trained on proprietary Dublin air quality data)

**Note**: `run_transfer_with_saved_data.py` uses simplified simulation since real source models are not publicly available. For exact thesis results matching [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), the full source models are required.

### 1. Baseline (No Transfer)
```python
from src.models.gp_model import BaselineGP

# Train on target city from scratch
model = BaselineGP(train_x, train_y)
model.train()
```

### 2. Prior Tempering Transfer
```python
from src.transfer_methods.prior_tempering import TemperedGP

# Transfer from Dublin to target city
target_model = TemperedGP(
    target_x, target_y,
    source_hyperparams=dublin_gp.hyperparams,
    beta=0.5  # Temperature parameter
)
```

### 3. Evaluation
```python
from src.evaluation.metrics import kl_divergence, picp

# Evaluate transfer quality
kl_div = kl_divergence(source_posterior, target_posterior)
coverage = picp(predictions, uncertainties, ground_truth)
```

## Experiments

### RQ1: Cross-Regional Generalization
Evaluates transfer from Dublin to other cities with different:
- Climate zones
- Urban morphology
- Pollution sources

### RQ2: Low-Cost Sensor Adaptation
Tests domain adaptation for:
- Sensor calibration drift
- Missing data scenarios
- Time-to-stabilization metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{oduori2025transfer,
  title={Probabilistic Transfer Learning for Air Quality Models},
  author={Oduori, Gabriel},
  year={2025},
  school={University}
}
```

## License

MIT License

## Contact

Gabriel Oduori - [GitHub](https://github.com/GabrielOduori)
