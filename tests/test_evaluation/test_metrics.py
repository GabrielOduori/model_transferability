"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
import torch
from scipy import stats
from src.evaluation.metrics import (
    regression_metrics,
    kl_divergence_distributions,
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    calibration_error,
    time_to_stabilization,
    transfer_efficiency,
    TransferEvaluator
)


class TestRegressionMetrics:
    """Tests for basic regression metrics."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        metrics = regression_metrics(y_true, y_pred)

        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-10)

    def test_constant_predictions(self):
        """Test metrics with constant predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        metrics = regression_metrics(y_true, y_pred)

        # R² should be 0 for constant predictions at mean
        assert metrics['r2'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0

    def test_inverse_predictions(self):
        """Test metrics with inverse predictions (worst case)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        metrics = regression_metrics(y_true, y_pred)

        # R² should be negative for worse than mean prediction
        assert metrics['r2'] < 0
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0

    def test_small_errors(self):
        """Test metrics with small random errors."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)

        metrics = regression_metrics(y_true, y_pred)

        assert 0 < metrics['r2'] < 1
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0

    def test_edge_case_single_sample(self):
        """Test with single sample."""
        y_true = np.array([5.0])
        y_pred = np.array([5.5])

        metrics = regression_metrics(y_true, y_pred)

        assert metrics['rmse'] == pytest.approx(0.5)
        assert metrics['mae'] == pytest.approx(0.5)
        # R² is undefined for single sample, but function should not crash


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_identical_distributions(self):
        """Test KL divergence between identical distributions."""
        np.random.seed(42)
        dist1 = np.random.randn(1000)
        dist2 = dist1.copy()

        kl_kde = kl_divergence_distributions(dist1, dist2, method='kde')

        # Should be close to 0
        assert kl_kde == pytest.approx(0.0, abs=0.1)

    def test_gaussian_closed_form(self):
        """Test Gaussian KL divergence with closed form."""
        np.random.seed(42)
        # N(0, 1)
        dist1 = np.random.randn(5000)
        # N(1, 1)
        dist2 = 1.0 + np.random.randn(5000)

        kl_gaussian = kl_divergence_distributions(dist1, dist2, method='gaussian')

        # Theoretical KL(N(0,1) || N(1,1)) = 0.5
        # (half the squared mean difference when variances are equal)
        assert kl_gaussian == pytest.approx(0.5, abs=0.2)

    def test_kde_method(self):
        """Test KL divergence with KDE method."""
        np.random.seed(42)
        dist1 = np.random.randn(1000)
        dist2 = 0.5 + np.random.randn(1000)

        kl_kde = kl_divergence_distributions(dist1, dist2, method='kde')

        # Should be positive (distributions are different)
        assert kl_kde > 0

    def test_different_variances(self):
        """Test KL divergence with different variances."""
        np.random.seed(42)
        # N(0, 1)
        dist1 = np.random.randn(2000)
        # N(0, 4)
        dist2 = 2.0 * np.random.randn(2000)

        kl_gaussian = kl_divergence_distributions(dist1, dist2, method='gaussian')

        # Should be positive
        assert kl_gaussian > 0

    def test_asymmetry(self):
        """Test that KL divergence is asymmetric."""
        np.random.seed(42)
        dist1 = np.random.randn(1000)
        dist2 = 1.0 + np.random.randn(1000)

        kl_12 = kl_divergence_distributions(dist1, dist2, method='kde')
        kl_21 = kl_divergence_distributions(dist2, dist1, method='kde')

        # KL is asymmetric: KL(P||Q) ≠ KL(Q||P)
        assert not np.isclose(kl_12, kl_21, rtol=0.1)

    def test_edge_case_constant_distribution(self):
        """Test KL divergence with constant values."""
        dist1 = np.ones(100) * 5.0
        dist2 = np.random.randn(100)

        # Should handle gracefully (may return inf or large value)
        kl = kl_divergence_distributions(dist1, dist2, method='kde')
        assert np.isfinite(kl) or np.isinf(kl)


class TestPredictionIntervalCoverage:
    """Tests for PICP (Prediction Interval Coverage Probability)."""

    def test_perfect_coverage(self):
        """Test PICP with perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true  # Perfect mean
        y_std = np.ones(n)  # True std is 1

        picp = prediction_interval_coverage_probability(
            y_true, y_pred, y_std, confidence=0.95
        )

        # Should be close to 0.95
        assert picp == pytest.approx(0.95, abs=0.05)

    def test_under_confident(self):
        """Test PICP with under-confident predictions (too wide intervals)."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true
        y_std = np.ones(n) * 10.0  # Overly large uncertainties

        picp = prediction_interval_coverage_probability(
            y_true, y_pred, y_std, confidence=0.95
        )

        # Should be higher than 0.95 (intervals too wide)
        assert picp > 0.95

    def test_over_confident(self):
        """Test PICP with over-confident predictions (too narrow intervals)."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true
        y_std = np.ones(n) * 0.1  # Overly small uncertainties

        picp = prediction_interval_coverage_probability(
            y_true, y_pred, y_std, confidence=0.95
        )

        # Should be lower than 0.95 (intervals too narrow)
        assert picp < 0.95

    def test_different_confidence_levels(self):
        """Test PICP at different confidence levels."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true
        y_std = np.ones(n)

        picp_68 = prediction_interval_coverage_probability(
            y_true, y_pred, y_std, confidence=0.68
        )
        picp_95 = prediction_interval_coverage_probability(
            y_true, y_pred, y_std, confidence=0.95
        )

        # Higher confidence should give higher coverage
        assert picp_95 > picp_68

    def test_zero_uncertainty(self):
        """Test PICP with zero uncertainty."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        y_std = np.zeros(5)

        picp = prediction_interval_coverage_probability(
            y_true, y_pred, y_std, confidence=0.95
        )

        # All points should be covered (interval width is 0, pred=true)
        assert picp == 1.0


class TestMeanPredictionIntervalWidth:
    """Tests for MPIW (Mean Prediction Interval Width)."""

    def test_constant_uncertainty(self):
        """Test MPIW with constant uncertainty."""
        y_std = np.ones(100)

        mpiw = mean_prediction_interval_width(y_std, confidence=0.95)

        # For N(0,1), 95% interval is approximately [-1.96, 1.96]
        expected = 2 * 1.96 * 1.0
        assert mpiw == pytest.approx(expected, rel=0.01)

    def test_varying_uncertainty(self):
        """Test MPIW with varying uncertainty."""
        y_std = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mpiw = mean_prediction_interval_width(y_std, confidence=0.95)

        # Should be mean of interval widths
        expected = 2 * 1.96 * np.mean(y_std)
        assert mpiw == pytest.approx(expected, rel=0.01)

    def test_different_confidence_levels(self):
        """Test MPIW at different confidence levels."""
        y_std = np.ones(100)

        mpiw_68 = mean_prediction_interval_width(y_std, confidence=0.68)
        mpiw_95 = mean_prediction_interval_width(y_std, confidence=0.95)

        # Higher confidence should give wider intervals
        assert mpiw_95 > mpiw_68

    def test_zero_uncertainty(self):
        """Test MPIW with zero uncertainty."""
        y_std = np.zeros(100)

        mpiw = mean_prediction_interval_width(y_std, confidence=0.95)

        assert mpiw == 0.0


class TestCalibrationError:
    """Tests for ECE (Expected Calibration Error)."""

    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000

        # Generate well-calibrated predictions
        y_true = np.random.randn(n)
        y_pred = y_true
        y_std = np.ones(n)

        ece = calibration_error(y_true, y_pred, y_std, n_bins=10)

        # Should be close to 0
        assert ece == pytest.approx(0.0, abs=0.1)

    def test_over_confident_calibration(self):
        """Test ECE with over-confident predictions."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true + 0.5 * np.random.randn(n)  # Add error
        y_std = np.ones(n) * 0.1  # Claim low uncertainty

        ece = calibration_error(y_true, y_pred, y_std, n_bins=10)

        # Should be high (poor calibration)
        assert ece > 0.1

    def test_under_confident_calibration(self):
        """Test ECE with under-confident predictions."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true
        y_std = np.ones(n) * 10.0  # Claim high uncertainty

        ece = calibration_error(y_true, y_pred, y_std, n_bins=10)

        # Should be moderate (conservative calibration)
        assert ece > 0.0

    def test_different_n_bins(self):
        """Test ECE with different number of bins."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.randn(n)
        y_pred = y_true + 0.2 * np.random.randn(n)
        y_std = np.ones(n)

        ece_5 = calibration_error(y_true, y_pred, y_std, n_bins=5)
        ece_20 = calibration_error(y_true, y_pred, y_std, n_bins=20)

        # Both should be finite and positive
        assert ece_5 >= 0
        assert ece_20 >= 0


class TestTimeToStabilization:
    """Tests for time-to-stabilization metric."""

    def test_immediate_stabilization(self):
        """Test when RMSE is immediately below threshold."""
        rmse_over_time = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        threshold = 0.1

        tts = time_to_stabilization(rmse_over_time, threshold)

        assert tts == 0  # Stable from the start

    def test_gradual_stabilization(self):
        """Test when RMSE gradually decreases."""
        rmse_over_time = np.array([0.5, 0.3, 0.2, 0.15, 0.08, 0.05])
        threshold = 0.1

        tts = time_to_stabilization(rmse_over_time, threshold)

        assert tts == 4  # First index where RMSE < 0.1

    def test_never_stabilizes(self):
        """Test when RMSE never reaches threshold."""
        rmse_over_time = np.array([0.5, 0.4, 0.3, 0.2, 0.15])
        threshold = 0.1

        tts = time_to_stabilization(rmse_over_time, threshold)

        assert tts == np.inf

    def test_fluctuating_rmse(self):
        """Test with fluctuating RMSE."""
        rmse_over_time = np.array([0.5, 0.08, 0.15, 0.05, 0.03])
        threshold = 0.1

        tts = time_to_stabilization(rmse_over_time, threshold)

        assert tts == 1  # First time it drops below threshold


class TestTransferEfficiency:
    """Tests for transfer efficiency metric."""

    def test_perfect_transfer(self):
        """Test with perfect transfer (baseline RMSE = transfer RMSE = 0)."""
        baseline_rmse = 0.0
        transfer_rmse = 0.0

        efficiency = transfer_efficiency(baseline_rmse, transfer_rmse)

        assert efficiency['improvement_pct'] == 0.0
        assert efficiency['sample_efficiency'] == 1.0

    def test_positive_transfer(self):
        """Test with positive transfer (improvement)."""
        baseline_rmse = 0.5
        transfer_rmse = 0.2

        efficiency = transfer_efficiency(baseline_rmse, transfer_rmse)

        expected_improvement = ((0.5 - 0.2) / 0.5) * 100
        assert efficiency['improvement_pct'] == pytest.approx(expected_improvement)
        assert efficiency['sample_efficiency'] > 1.0

    def test_negative_transfer(self):
        """Test with negative transfer (degradation)."""
        baseline_rmse = 0.2
        transfer_rmse = 0.5

        efficiency = transfer_efficiency(baseline_rmse, transfer_rmse)

        # Improvement should be negative
        assert efficiency['improvement_pct'] < 0
        assert efficiency['sample_efficiency'] < 1.0

    def test_no_improvement(self):
        """Test with no improvement."""
        baseline_rmse = 0.3
        transfer_rmse = 0.3

        efficiency = transfer_efficiency(baseline_rmse, transfer_rmse)

        assert efficiency['improvement_pct'] == 0.0
        assert efficiency['sample_efficiency'] == 1.0


class TestTransferEvaluator:
    """Tests for TransferEvaluator class."""

    def test_initialization(self):
        """Test TransferEvaluator initialization."""
        evaluator = TransferEvaluator()
        assert evaluator is not None

    def test_evaluate_rq1(self, sample_predictions, sample_distributions):
        """Test RQ1 evaluation (cross-regional generalization)."""
        evaluator = TransferEvaluator()

        y_true = sample_predictions['y_true']
        y_pred_baseline = sample_predictions['y_pred']
        y_std_baseline = sample_predictions['y_std']

        y_pred_transfer = y_pred_baseline - 0.05  # Slightly better
        y_std_transfer = y_std_baseline * 0.9

        source_preds = sample_distributions['dist1']
        target_preds = sample_distributions['dist2']

        metrics = evaluator.evaluate_rq1(
            y_true=y_true,
            y_pred_baseline=y_pred_baseline,
            y_std_baseline=y_std_baseline,
            y_pred_transfer=y_pred_transfer,
            y_std_transfer=y_std_transfer,
            source_predictions=source_preds,
            target_predictions=target_preds
        )

        # Check all expected keys are present
        assert 'kl_divergence' in metrics
        assert 'baseline_rmse' in metrics
        assert 'baseline_mae' in metrics
        assert 'baseline_r2' in metrics
        assert 'transfer_rmse' in metrics
        assert 'transfer_mae' in metrics
        assert 'transfer_r2' in metrics
        assert 'baseline_picp' in metrics
        assert 'transfer_picp' in metrics
        assert 'baseline_ece' in metrics
        assert 'transfer_ece' in metrics

    def test_evaluate_rq2(self, sample_predictions):
        """Test RQ2 evaluation (sensor adaptation)."""
        evaluator = TransferEvaluator()

        y_true = sample_predictions['y_true']
        y_pred_baseline = sample_predictions['y_pred']
        y_std_baseline = sample_predictions['y_std']

        y_pred_transfer = y_pred_baseline - 0.05
        y_std_transfer = y_std_baseline * 0.9

        # Simulate RMSE over time
        baseline_rmse_over_time = np.array([0.5, 0.4, 0.3, 0.25, 0.2, 0.18])
        transfer_rmse_over_time = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07])

        metrics = evaluator.evaluate_rq2(
            y_true=y_true,
            y_pred_baseline=y_pred_baseline,
            y_std_baseline=y_std_baseline,
            y_pred_transfer=y_pred_transfer,
            y_std_transfer=y_std_transfer,
            baseline_rmse_over_time=baseline_rmse_over_time,
            transfer_rmse_over_time=transfer_rmse_over_time,
            stabilization_threshold=0.15
        )

        # Check expected keys
        assert 'baseline_time_to_stab' in metrics
        assert 'transfer_time_to_stab' in metrics
        assert 'transfer_efficiency' in metrics

    def test_print_summary(self, sample_predictions, sample_distributions, capsys):
        """Test print_summary method."""
        evaluator = TransferEvaluator()

        y_true = sample_predictions['y_true']
        y_pred_baseline = sample_predictions['y_pred']
        y_std_baseline = sample_predictions['y_std']

        y_pred_transfer = y_pred_baseline - 0.05
        y_std_transfer = y_std_baseline * 0.9

        source_preds = sample_distributions['dist1']
        target_preds = sample_distributions['dist2']

        metrics = evaluator.evaluate_rq1(
            y_true=y_true,
            y_pred_baseline=y_pred_baseline,
            y_std_baseline=y_std_baseline,
            y_pred_transfer=y_pred_transfer,
            y_std_transfer=y_std_transfer,
            source_predictions=source_preds,
            target_predictions=target_preds
        )

        evaluator.print_summary(metrics)

        # Capture printed output
        captured = capsys.readouterr()
        assert 'KL Divergence' in captured.out or len(captured.out) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
