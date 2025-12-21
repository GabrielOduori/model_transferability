# Implementation Guide: Transfer Learning for Air Quality Models

This guide explains how to use the transfer learning framework for your thesis research.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Implementation Details](#implementation-details)
5. [Experimental Workflow](#experimental-workflow)
6. [Mapping to Thesis Chapter](#mapping-to-thesis-chapter)

## Overview

This repository implements three probabilistic transfer learning paradigms:

### 1. Bayesian Prior Transfer with Tempering ✅ **IMPLEMENTED**

**What it does**: Uses the posterior distribution from Dublin as a "soft prior" for the target city, weighted by temperature β.

**Formula**:
```
p(θ | D_T, D_S) ∝ p(D_T | θ) · [p(θ | D_S)]^β
```

**When to use**:
- Transferring GP models between cities
- You have a trained source model and limited target data
- You want to control how much you trust source knowledge

**Implementation**: `src/transfer_methods/prior_tempering.py`

### 2. Optimal Bayesian Transfer Learning (OBTL) ⚠️ **PENDING**

**What it does**: Models source and target parameters as coming from a joint distribution (not just sequential update).

**Status**: Requires clarification on whether your GP uses:
- Fixed spatial grid → Can use Wishart priors
- Continuous kernel → Need hyperparameter transfer (simpler)

**Recommendation**: Start with Prior Tempering (already implemented). OBTL adds theoretical rigor but is more complex.

### 3. Deep Probabilistic Transfer Regression (DPTR) ⚠️ **PENDING**

**What it does**: Uses VAE to align land-use features between cities.

**When to use**: For GAL-SSM-LUR model with missing data scenarios

**Status**: Awaiting your GAL-SSM-LUR model architecture details

## Quick Start

### Installation

```bash
cd model_transferability
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Demo Experiment

```bash
cd model_transferability
python experiments/demo_prior_tempering.py
```

This will:
1. Generate synthetic data (Dublin → Cork transfer)
2. Train source model
3. Compare transfer strategies (β = 0.0, 0.3, 0.5, 0.7, 1.0)
4. Generate visualizations
5. Save results to `results/figures/`

### Use in Your Own Code

```python
from src.models.gp_model import BaselineGP, train_baseline_gp
from src.transfer_methods.prior_tempering import train_tempered_gp
from src.evaluation.metrics import TransferEvaluator

# 1. Train source model (Dublin)
source_model, source_likelihood = train_baseline_gp(
    model, likelihood, dublin_x, dublin_y
)

# 2. Transfer to target city with β=0.5
target_model, target_likelihood = train_tempered_gp(
    source_gp=source_model,
    target_x=cork_x,
    target_y=cork_y,
    beta=0.5
)

# 3. Evaluate
evaluator = TransferEvaluator()
results = evaluator.evaluate_rq1(
    source_posterior, target_posterior,
    predictions, uncertainties, true_values
)
```

## Core Concepts

### Temperature Parameter β

The temperature parameter controls the transfer:

| β Value | Meaning | Use Case |
|---------|---------|----------|
| β = 0.0 | Ignore source completely | Baseline (no transfer) |
| β = 0.3 | Weak source influence | Very different cities |
| β = 0.5 | Balanced | Similar climate, different topography |
| β = 0.7 | Strong source influence | Very similar cities |
| β = 1.0 | Full source knowledge | Nearly identical domains |

**How to choose β**:
- Run experiments with multiple β values
- Select based on validation RMSE or KL divergence
- Can also optimize β as a hyperparameter

### KL Divergence (RQ1 Metric)

**What it measures**: How different the target posterior is from the source posterior.

**Interpretation**:
- Low KL → Successful transfer (domains are similar)
- High KL → Poor transfer (domains are very different)

**Usage**:
```python
from src.evaluation.metrics import kl_divergence_distributions

# Sample posteriors
source_samples = sample_posterior(source_gp, n_samples=1000)
target_samples = sample_posterior(target_gp, n_samples=1000)

# Compute KL divergence
kl_div = kl_divergence_distributions(source_samples, target_samples)
print(f"Domain distance: {kl_div:.4f}")
```

### PICP (RQ2 Metric)

**What it measures**: Prediction Interval Coverage Probability - the proportion of observations within confidence bounds.

**Interpretation**:
- PICP ≈ 0.95 → Well-calibrated uncertainties
- PICP < 0.95 → Overconfident (intervals too narrow)
- PICP > 0.95 → Underconfident (intervals too wide)

**Usage**:
```python
from src.evaluation.metrics import prediction_interval_coverage_probability

picp = prediction_interval_coverage_probability(
    predictions, uncertainties, true_values, confidence=0.95
)
print(f"Coverage: {picp:.2%} (target: 95%)")
```

## Implementation Details

### File Structure

```
src/
├── models/
│   └── gp_model.py              # BaselineGP, SpatialTemporalGP
├── transfer_methods/
│   ├── prior_tempering.py       # ✅ TemperedGP, train_tempered_gp
│   ├── obtl.py                  # ⚠️ TODO: OBTL implementation
│   └── dptr.py                  # ⚠️ TODO: DPTR for GAL-SSM-LUR
├── evaluation/
│   └── metrics.py               # ✅ All evaluation metrics
└── data/
    └── loaders.py               # TODO: Data loading utilities
```

### Key Classes

#### `TemperedGP` (Prior Tempering)

```python
class TemperedGP(gpytorch.models.ExactGP):
    """GP with tempered prior from source domain."""

    def __init__(self, train_x, train_y, likelihood,
                 source_hyperparams, beta=1.0):
        # Initializes with source hyperparameters
        # beta controls prior strength
```

**Parameters to transfer**:
- Kernel lengthscales (spatial correlation)
- Kernel outputscale (signal variance)
- Mean function constant (baseline pollution level)
- Observation noise (measurement uncertainty)

#### `TransferEvaluator` (Comprehensive Evaluation)

```python
evaluator = TransferEvaluator(confidence=0.95)

# For RQ1: Cross-regional generalization
rq1_results = evaluator.evaluate_rq1(
    source_posterior_samples,
    target_posterior_samples,
    predictions, uncertainties, true_values
)

# For RQ2: Sensor adaptation
rq2_results = evaluator.evaluate_rq2(
    rmse_over_time, target_rmse,
    predictions, uncertainties, true_values
)

evaluator.print_summary()
```

## Experimental Workflow

### RQ1: Cross-Regional Generalization

**Objective**: Evaluate how well Dublin models transfer to different cities.

**Experimental Design**:

```python
# 1. Select target cities with varying similarity to Dublin
target_cities = {
    'cork': {'climate': 'similar', 'topography': 'similar'},
    'barcelona': {'climate': 'different', 'topography': 'coastal'},
    'milan': {'climate': 'different', 'topography': 'valley'}
}

# 2. For each target city:
for city_name, characteristics in target_cities.items():
    # Train source model (Dublin)
    source_model = train_on_dublin(dublin_data)

    # Transfer with multiple β values
    for beta in [0.0, 0.3, 0.5, 0.7, 1.0]:
        target_model = train_tempered_gp(
            source_model, target_data, beta=beta
        )

        # Evaluate
        metrics = evaluate_rq1(target_model, test_data)
        results[city_name][beta] = metrics

# 3. Analyze:
#    - Which β works best for each city type?
#    - Does KL divergence correlate with city similarity?
#    - How much data reduction vs. baseline?
```

**Success Criteria** (from your thesis):
- **KL Divergence** reduced compared to baseline
- **RMSE** comparable or better than baseline with less data
- **PICP** remains ≈ 0.95 (uncertainty is calibrated)

### RQ2: Sensor Adaptation

**Objective**: Evaluate rapid calibration of low-cost sensors in new deployments.

**Experimental Design**:

```python
# 1. Simulate sensor deployment
# Start with uncalibrated sensor, collect data over time

time_steps = range(1, 100)  # Days since deployment
rmse_baseline = []
rmse_transfer = []

for t in time_steps:
    # Get data up to time t
    data_t = get_data_up_to_time(t)

    # Baseline: train from scratch
    baseline_model = train_baseline_gp(data_t)
    rmse_baseline.append(evaluate(baseline_model))

    # Transfer: use Dublin as prior
    transfer_model = train_tempered_gp(
        source_gp=dublin_model,
        target_x=data_t.X,
        target_y=data_t.y,
        beta=0.5
    )
    rmse_transfer.append(evaluate(transfer_model))

# 2. Compute time-to-stabilization
target_rmse = 2.0  # Acceptable RMSE threshold
t_baseline = time_to_stabilization(rmse_baseline, target_rmse)
t_transfer = time_to_stabilization(rmse_transfer, target_rmse)

speedup = (t_baseline - t_transfer) / t_baseline * 100
print(f"Transfer stabilizes {speedup:.0f}% faster")
```

**Success Criteria** (from your thesis):
- **60-80% faster stabilization** with transfer
- **PICP ≈ 0.95** throughout adaptation period
- **Lower RMSE** at all time steps compared to baseline

## Mapping to Thesis Chapter

### Your Thesis Structure → Code Mapping

| Thesis Section | Implementation Status | Files |
|----------------|----------------------|-------|
| **3.1 Prior Tempering** | ✅ Complete | `src/transfer_methods/prior_tempering.py` |
| **3.2 OBTL** | ⚠️ Needs clarification | `src/transfer_methods/obtl.py` (TODO) |
| **3.3 DPTR** | ⚠️ Needs GAL-SSM-LUR | `src/transfer_methods/dptr.py` (TODO) |
| **4.1 RQ1 Experiments** | ✅ Framework ready | `experiments/rq1_cross_regional.py` (TODO) |
| **4.2 RQ2 Experiments** | ⚠️ Needs DPTR | `experiments/rq2_sensor_adaptation.py` (TODO) |
| **Evaluation Metrics** | ✅ Complete | `src/evaluation/metrics.py` |

### Figures for Thesis

The code generates publication-ready figures:

1. **Figure 7.X: Transfer Performance vs β**
   - Shows RMSE for different temperature values
   - Demonstrates optimal β selection

2. **Figure 7.X: Domain Distance (KL Divergence)**
   - Shows how KL divergence varies with β
   - Answers: "How similar are the domains after transfer?"

3. **Figure 7.X: Prediction Scatter Plots**
   - True vs. predicted values
   - Visual assessment of transfer quality

4. **Figure 7.X: Uncertainty Calibration**
   - PICP across different β values
   - Shows reliability of transferred uncertainties

## Next Steps for Your Research

### Immediate (Can do now)

1. ✅ Run `demo_prior_tempering.py` to see the framework in action
2. ✅ Modify synthetic data to match your Dublin data characteristics
3. ✅ Replace synthetic data with real Dublin data
4. ✅ Test transfer to a real target city (even with limited data)

### Short-term (Need your input)

1. **Provide Dublin model details**:
   - What features are in your GP?
   - What pollutants are you predicting?
   - Do you have trained model weights?

2. **Specify target city**:
   - Which city are you transferring to?
   - How much data is available?
   - What's the sensor setup?

3. **GAL-SSM-LUR architecture**:
   - Share model structure
   - Explain state-space formulation
   - Identify which components need DPTR

### Long-term (Full thesis implementation)

1. Implement OBTL (if you decide it's needed)
2. Implement DPTR for GAL-SSM-LUR
3. Run full RQ1 experiments with multiple cities
4. Run full RQ2 experiments with temporal adaptation
5. Generate all thesis figures and tables

## Troubleshooting

### Common Issues

**Issue**: "RuntimeError: Sizes of tensors must match"
- **Cause**: Input dimensions don't match between source and target
- **Solution**: Ensure both cities have same feature set

**Issue**: "Negative transfer" (target worse than baseline)
- **Cause**: β too high for dissimilar domains
- **Solution**: Reduce β or check domain similarity

**Issue**: "PICP far from 0.95"
- **Cause**: Poorly calibrated uncertainties
- **Solution**: Adjust likelihood noise parameter or prior variance

## Questions?

For implementation questions or thesis guidance, please:
1. Check this guide first
2. Review code docstrings
3. Run the demo experiment
4. Open an issue on GitHub

Good luck with your research!
