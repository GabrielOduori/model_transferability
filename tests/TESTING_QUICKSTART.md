# Testing Quick Start Guide

## TL;DR

```bash
# Install dependencies
pip install pytest pytest-cov

# Run all tests
./run_tests.sh all

# Or
pytest

# Check coverage
./run_tests.sh coverage
```

**Result**: 112 tests covering models, transfer methods, metrics, and integration

---

## What's Tested?

### ✅ Models (30+ tests)
- **BaselineGP**: Standard GP baseline
- **SpatialTemporalGP**: Spatio-temporal GP
- Training, predictions, hyperparameters

### ✅ Transfer Methods (60+ tests)
- **Prior Tempering**: Bayesian prior tempering (17 tests)
- **OBTL**: Optimal Bayesian Transfer Learning (20 tests)
- **DPTR**: Deep Probabilistic Transfer Regression (25 tests)

### ✅ Evaluation Metrics (35+ tests)
- RMSE, MAE, R²
- KL Divergence
- PICP, MPIW, ECE
- Transfer efficiency

### ✅ Integration (9+ tests)
- End-to-end pipelines
- Method comparisons
- Real-world workflows

---

## File Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── test_models/
│   └── test_gp_model.py            # 30+ tests
├── test_transfer_methods/
│   ├── test_prior_tempering.py     # 17 tests
│   ├── test_obtl.py                # 20 tests
│   └── test_dptr.py                # 25 tests
├── test_evaluation/
│   └── test_metrics.py             # 35+ tests
└── test_integration/
    └── test_full_pipelines.py      # 9 tests
```

---

## Common Commands

```bash
# All tests
pytest

# Specific category
pytest tests/test_models/
pytest tests/test_transfer_methods/
pytest tests/test_evaluation/
pytest tests/test_integration/

# Specific file
pytest tests/test_models/test_gp_model.py

# Specific test
pytest tests/test_models/test_gp_model.py::TestBaselineGP::test_initialization_without_ard

# With coverage
pytest --cov=src --cov-report=html

# Verbose
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

---

## Using the Test Runner

```bash
chmod +x run_tests.sh  # First time only

./run_tests.sh all          # All tests
./run_tests.sh unit         # Unit tests only
./run_tests.sh integration  # Integration tests
./run_tests.sh coverage     # With coverage report
./run_tests.sh quick        # Fast tests only
./run_tests.sh help         # Show help
```

---

## What Each Test File Does

### `test_gp_model.py`
Tests GP model initialization, training, and predictions

**Key tests:**
- Model initialization (with/without ARD)
- Forward passes
- Training convergence
- Uncertainty quantification

### `test_prior_tempering.py`
Tests prior tempering transfer learning

**Key tests:**
- Tempered MLL computation
- Beta parameter effects
- Hyperparameter transfer
- Full transfer pipeline

### `test_obtl.py`
Tests OBTL transfer learning

**Key tests:**
- Inducing point selection
- Covariance transfer
- Wishart weighting
- Delta parameter effects

### `test_dptr.py`
Tests DPTR with VAE feature alignment

**Key tests:**
- Encoder/decoder networks
- VAE training
- Feature alignment
- Latent space mapping

### `test_metrics.py`
Tests all evaluation metrics

**Key tests:**
- Regression metrics (RMSE, MAE, R²)
- KL divergence
- Calibration metrics (PICP, ECE)
- Transfer efficiency

### `test_full_pipelines.py`
Integration tests for complete workflows

**Key tests:**
- End-to-end pipelines
- Method comparisons
- Real transfer scenarios

---

## Example Test Run

```bash
$ pytest tests/test_models/test_gp_model.py -v

tests/test_models/test_gp_model.py::TestBaselineGP::test_initialization_without_ard PASSED
tests/test_models/test_gp_model.py::TestBaselineGP::test_initialization_with_ard PASSED
tests/test_models/test_gp_model.py::TestBaselineGP::test_forward_pass PASSED
tests/test_models/test_gp_model.py::TestBaselineGP::test_training_convergence PASSED
...

===== 30 passed in 15.23s =====
```

---

## Coverage Report

After running with coverage:

```bash
./run_tests.sh coverage

# Open in browser
open htmlcov/index.html
```

**Expected coverage:**
- Models: > 90%
- Transfer methods: > 90%
- Metrics: > 85%
- Overall: > 80%

---

## Fixtures Available

All tests have access to these fixtures (from `conftest.py`):

### Data
- `simple_1d_data` - 1D regression data
- `simple_2d_data` - 2D regression data
- `spatiotemporal_data` - Spatial + temporal features
- `source_target_1d` - Transfer learning data (1D)
- `source_target_2d` - Transfer learning data (2D)
- `feature_mismatch_data` - Different dimensions for DPTR

### Utilities
- `device` - Auto CUDA/CPU selection
- `random_seed` - Reproducible randomness
- `trained_likelihood` - Pre-initialized likelihood
- `mock_trained_model` - Pre-trained model

---

## Writing a New Test

```python
# tests/test_your_module/test_your_feature.py

import pytest
import torch
from src.your_module import YourClass

class TestYourClass:
    """Tests for YourClass."""

    def test_basic_functionality(self, simple_1d_data, device):
        """Test that basic functionality works."""
        # Arrange
        x, y = simple_1d_data
        x = x.to(device)
        y = y.to(device)

        # Act
        model = YourClass()
        result = model(x)

        # Assert
        assert result.shape == y.shape
        assert torch.isfinite(result).all()
```

---

## Debugging Failed Tests

```bash
# Full traceback
pytest --tb=long

# Drop to debugger on failure
pytest --pdb

# Show local variables
pytest -l

# Rerun failed tests
pytest --lf
```

---

## CI/CD Ready

Tests are configured for continuous integration:

- `pytest.ini` - Test configuration
- Coverage reporting (HTML, XML, terminal)
- Markers for slow/GPU tests
- Parallel execution support

---

## Next Steps

1. ✅ **Run tests**: `./run_tests.sh all`
2. ✅ **Check coverage**: `./run_tests.sh coverage`
3. ✅ **Read examples**: Browse test files
4. ✅ **Add your tests**: Follow existing patterns
5. ✅ **Keep green**: Run before committing

---

## Documentation

- [tests/README.md](README.md) - Detailed test documentation
- [tests/TEST_SUMMARY.md](TEST_SUMMARY.md) - Complete test breakdown
- [../TESTING.md](../TESTING.md) - Comprehensive testing guide

---

## Test Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 112 |
| Test Files | 6 |
| Lines of Test Code | ~3000 |
| Coverage Target | > 80% |
| Avg Test Runtime | < 1 min |

---

**Questions?** Check the full documentation or run `./run_tests.sh help`
