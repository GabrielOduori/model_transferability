# Test Suite for Model Transferability

This directory contains comprehensive tests for the model transferability project.

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and test configuration
├── test_models/                     # Tests for GP models
│   └── test_gp_model.py            # BaselineGP and SpatialTemporalGP tests
├── test_transfer_methods/           # Tests for transfer learning methods
│   ├── test_prior_tempering.py     # Prior tempering transfer tests
│   ├── test_obtl.py                # OBTL transfer tests
│   └── test_dptr.py                # DPTR transfer tests
├── test_evaluation/                 # Tests for evaluation metrics
│   └── test_metrics.py             # Metrics and evaluator tests
└── test_integration/                # Integration tests
    └── test_full_pipelines.py      # End-to-end pipeline tests
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/test_models tests/test_transfer_methods tests/test_evaluation

# Run only integration tests
pytest tests/test_integration

# Run tests for a specific module
pytest tests/test_models/test_gp_model.py

# Run a specific test class
pytest tests/test_models/test_gp_model.py::TestBaselineGP

# Run a specific test function
pytest tests/test_models/test_gp_model.py::TestBaselineGP::test_initialization_without_ard
```

### Run Tests with Markers

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only GPU tests (if GPU available)
pytest -m gpu
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest -n 4
```

## Test Coverage

### View Coverage Report

After running tests with coverage:

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=src --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=src --cov-report=xml
```

### Coverage Goals

- **Overall coverage**: > 80%
- **Critical modules** (models, transfer methods): > 90%
- **Utility functions**: > 70%

## Test Categories

### Unit Tests

Test individual components in isolation:

- **Model Tests** (`test_models/`): GP model initialization, forward passes, training
- **Transfer Method Tests** (`test_transfer_methods/`): Individual transfer algorithms
- **Metric Tests** (`test_evaluation/`): Evaluation metrics and calculations

### Integration Tests

Test complete workflows:

- **Pipeline Tests** (`test_integration/`): End-to-end transfer learning pipelines
- **Multi-method Comparisons**: Compare different transfer methods on same data

## Fixtures

Common fixtures are defined in `conftest.py`:

### Data Fixtures

- `simple_1d_data`: Basic 1D regression data
- `simple_2d_data`: Basic 2D regression data
- `spatiotemporal_data`: Spatio-temporal data (spatial + temporal dims)
- `source_target_1d`: Source and target domains for transfer (1D)
- `source_target_2d`: Source and target domains for transfer (2D)
- `feature_mismatch_data`: Different feature dimensions (for DPTR)

### Model Fixtures

- `trained_likelihood`: Pre-initialized likelihood
- `mock_trained_model`: Mock trained GP model
- `sample_predictions`: Sample predictions for metrics testing
- `sample_distributions`: Sample distributions for KL divergence

### Device Fixture

- `device`: Automatically selects CUDA if available, else CPU

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<functionality>`

### Example Test Structure

```python
import pytest
import torch
from src.models.your_model import YourModel

class TestYourModel:
    """Tests for YourModel class."""

    def test_initialization(self, device):
        """Test model initialization."""
        model = YourModel()
        assert model is not None

    def test_forward_pass(self, simple_1d_data, device):
        """Test forward pass."""
        x, y = simple_1d_data
        x = x.to(device)
        y = y.to(device)

        model = YourModel()
        output = model(x)

        assert output.shape == y.shape
        assert torch.isfinite(output).all()
```

### Using Fixtures

```python
def test_with_fixture(self, source_target_1d, device):
    """Test using source/target data fixture."""
    x_source, y_source = source_target_1d['source']
    x_target, y_target = source_target_1d['target']

    # Your test code here
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_dim", [1, 2, 5, 10])
def test_different_dimensions(self, input_dim, device):
    """Test with different input dimensions."""
    x = torch.randn(50, input_dim).to(device)
    # Test code
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Use descriptive test names and docstrings
3. **Coverage**: Test normal cases, edge cases, and error conditions
4. **Speed**: Keep unit tests fast; mark slow tests with `@pytest.mark.slow`
5. **Assertions**: Use specific assertions with meaningful messages
6. **Fixtures**: Reuse fixtures to avoid duplication
7. **Clean up**: Use fixtures for setup/teardown, not manual cleanup

## Common Patterns

### Testing Convergence

```python
def test_training_convergence(self, simple_1d_data, device):
    """Test that training decreases loss."""
    # Train model
    model, likelihood, losses = train_model(...)

    # Check convergence
    assert losses[-1] < losses[0]
    assert np.isfinite(losses).all()
```

### Testing Predictions

```python
def test_prediction_shape(self, model, test_data, device):
    """Test prediction output shape."""
    x_test, y_test = test_data

    predictions = model.predict(x_test)

    assert predictions.shape == y_test.shape
    assert torch.isfinite(predictions).all()
```

### Testing Hyperparameters

```python
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_beta_parameter(self, beta, device):
    """Test with different beta values."""
    model = train_with_beta(beta=beta)
    assert model is not None
```

## Debugging Failed Tests

### Run with Debug Info

```bash
# Show full diff on assertion errors
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Run only failed tests from last run
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Common Issues

1. **CUDA out of memory**: Reduce batch sizes or use CPU
2. **Random failures**: Set random seeds in fixtures
3. **Numerical precision**: Use `pytest.approx()` for float comparisons
4. **Fixture not found**: Check `conftest.py` import

## Continuous Integration

Tests should run automatically on:

- Every commit (unit tests)
- Every pull request (unit + integration tests)
- Nightly builds (full test suite with slow tests)

## Test Metrics

Track these metrics over time:

- Test coverage percentage
- Number of tests
- Test execution time
- Failure rate

## Contributing

When adding new features:

1. Write tests first (TDD) or alongside implementation
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=src --cov-report=term-missing`
4. Add docstrings to test functions
5. Update this README if adding new test categories

## FAQ

**Q: Tests are slow. How can I speed them up?**

A: Use `pytest -n auto` for parallel execution, or run specific test files.

**Q: How do I test GPU-specific functionality?**

A: Use the `device` fixture and mark tests with `@pytest.mark.gpu`.

**Q: Tests pass locally but fail in CI. Why?**

A: Check random seeds, file paths, and environment dependencies.

**Q: How do I test code that uses external packages (gam_ssm_lur, fusiongp)?**

A: Mark with `@pytest.mark.optional` and use `pytest.importorskip()`.

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing PyTorch Models](https://pytorch.org/tutorials/beginner/basics/testing.html)
