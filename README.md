# Air Quality Model Transferability

This repository implements transfer learning methods for air quality models across different geographic regions, supporting the research presented in the thesis chapter on probabilistic transfer learning.

## Research Questions

**RQ1**: How can transfer learning be used to generalize air-quality models across different geographic regions?

**RQ2**: What domain-adaptation techniques are most effective for low-cost sensor networks?

## Overview

This project implements three probabilistic transfer learning paradigms:

1. **Bayesian Prior Transfer with Tempering** - Adaptive knowledge transfer using temperature parameter β
2. **Optimal Bayesian Transfer Learning (OBTL)** - Joint Wishart priors for covariance structure transfer
3. **Deep Probabilistic Transfer Regression (DPTR)** - VAE-based domain adaptation for missing data scenarios

## Models

- **GPyTorch Gaussian Process**: Spatial-temporal air quality prediction with uncertainty quantification
- **GAL-SSM-LUR**: Geographically Adaptive LASSO State Space Model with Land Use Regression

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
