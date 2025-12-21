"""
Evaluation Metrics for Transfer Learning

Implements metrics for assessing transfer learning quality:
- KL Divergence for RQ1 (domain distance)
- PICP for RQ2 (uncertainty calibration)
- Standard regression metrics (RMSE, MAE, R²)
"""

import torch
import numpy as np
from typing import Tuple, Optional
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def kl_divergence_distributions(
    source_posterior_samples: np.ndarray,
    target_posterior_samples: np.ndarray,
    method: str = 'kde'
) -> float:
    """
    Compute KL divergence between source and target posterior distributions.

    KL(p_target || p_source) = E_target[log(p_target / p_source)]

    A lower KL divergence indicates successful knowledge transfer, as the
    target posterior is closer to the source posterior.

    Parameters
    ----------
    source_posterior_samples : np.ndarray
        Samples from source posterior p(θ|D_S) [N_s, D]
    target_posterior_samples : np.ndarray
        Samples from target posterior p(θ|D_T) [N_t, D]
    method : str, default='kde'
        Method for density estimation ('kde' or 'gaussian')

    Returns
    -------
    float
        KL divergence estimate

    Examples
    --------
    >>> # Sample from posteriors
    >>> source_samples = sample_posterior(source_gp, n_samples=1000)
    >>> target_samples = sample_posterior(target_gp, n_samples=1000)
    >>>
    >>> # Compute KL divergence
    >>> kl_div = kl_divergence_distributions(source_samples, target_samples)
    >>> print(f"Domain distance: {kl_div:.4f}")
    """
    if method == 'kde':
        # Kernel Density Estimation
        source_kde = gaussian_kde(source_posterior_samples.T)
        target_kde = gaussian_kde(target_posterior_samples.T)

        # Monte Carlo estimate
        log_p_target = target_kde.logpdf(target_posterior_samples.T)
        log_p_source = source_kde.logpdf(target_posterior_samples.T)

        kl = np.mean(log_p_target - log_p_source)

    elif method == 'gaussian':
        # Assume Gaussian distributions
        source_mean = np.mean(source_posterior_samples, axis=0)
        source_cov = np.cov(source_posterior_samples.T)

        target_mean = np.mean(target_posterior_samples, axis=0)
        target_cov = np.cov(target_posterior_samples.T)

        # Closed-form KL for Gaussians
        k = len(source_mean)
        inv_source_cov = np.linalg.inv(source_cov)

        kl = 0.5 * (
            np.trace(inv_source_cov @ target_cov) +
            (source_mean - target_mean).T @ inv_source_cov @ (source_mean - target_mean) -
            k +
            np.log(np.linalg.det(source_cov) / np.linalg.det(target_cov))
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(kl)


def prediction_interval_coverage_probability(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    true_values: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Compute Prediction Interval Coverage Probability (PICP).

    PICP measures the proportion of observations that fall within the
    predicted confidence intervals. A well-calibrated model should have
    PICP ≈ confidence level.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions [N]
    uncertainties : np.ndarray
        Prediction standard deviations [N]
    true_values : np.ndarray
        Ground truth observations [N]
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% intervals)

    Returns
    -------
    float
        PICP score (should be ≈ confidence if well-calibrated)

    Examples
    --------
    >>> # Make predictions with uncertainty
    >>> preds, stds = model.predict(test_x)
    >>>
    >>> # Compute PICP
    >>> picp = prediction_interval_coverage_probability(
    ...     preds, stds, test_y, confidence=0.95
    ... )
    >>> print(f"Coverage: {picp:.2%} (target: 95%)")
    """
    from scipy.stats import norm

    # Compute z-score for confidence level
    z_score = norm.ppf((1 + confidence) / 2)

    # Compute confidence bounds
    lower_bound = predictions - z_score * uncertainties
    upper_bound = predictions + z_score * uncertainties

    # Check coverage
    coverage = np.mean(
        (true_values >= lower_bound) & (true_values <= upper_bound)
    )

    return float(coverage)


def mean_prediction_interval_width(
    uncertainties: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Compute mean prediction interval width.

    Narrower intervals (with maintained coverage) indicate more certain
    predictions.

    Parameters
    ----------
    uncertainties : np.ndarray
        Prediction standard deviations [N]
    confidence : float, default=0.95
        Confidence level

    Returns
    -------
    float
        Mean interval width
    """
    from scipy.stats import norm

    z_score = norm.ppf((1 + confidence) / 2)
    interval_width = 2 * z_score * uncertainties

    return float(np.mean(interval_width))


def calibration_error(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    true_values: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for regression.

    Bins predictions by confidence and compares expected vs. observed coverage.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions [N]
    uncertainties : np.ndarray
        Prediction standard deviations [N]
    true_values : np.ndarray
        Ground truth observations [N]
    n_bins : int, default=10
        Number of confidence bins

    Returns
    -------
    float
        Expected calibration error
    """
    from scipy.stats import norm

    # Compute normalized residuals (should be ~ N(0,1) if calibrated)
    residuals = (true_values - predictions) / uncertainties
    abs_residuals = np.abs(residuals)

    # Expected vs. observed coverage at different confidence levels
    confidence_levels = np.linspace(0.1, 0.9, n_bins)
    ece = 0.0

    for conf in confidence_levels:
        z_score = norm.ppf((1 + conf) / 2)

        # Expected: conf% should be within z_score std devs
        expected_coverage = conf

        # Observed: what proportion actually is?
        observed_coverage = np.mean(abs_residuals <= z_score)

        # Accumulate error
        ece += np.abs(expected_coverage - observed_coverage) / n_bins

    return float(ece)


def regression_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray
) -> dict:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions [N]
    true_values : np.ndarray
        Ground truth observations [N]

    Returns
    -------
    dict
        Dictionary with RMSE, MAE, R² metrics
    """
    return {
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'r2': r2_score(true_values, predictions)
    }


def transfer_efficiency(
    baseline_rmse: float,
    transfer_rmse: float,
    n_target_samples: int
) -> dict:
    """
    Compute transfer learning efficiency metrics.

    Parameters
    ----------
    baseline_rmse : float
        RMSE of model trained from scratch on target data
    transfer_rmse : float
        RMSE of transferred model on target data
    n_target_samples : int
        Number of target domain samples used

    Returns
    -------
    dict
        Transfer efficiency metrics
    """
    improvement = (baseline_rmse - transfer_rmse) / baseline_rmse * 100
    sample_efficiency = baseline_rmse / (transfer_rmse * n_target_samples)

    return {
        'improvement_percent': improvement,
        'sample_efficiency': sample_efficiency,
        'baseline_rmse': baseline_rmse,
        'transfer_rmse': transfer_rmse
    }


def time_to_stabilization(
    rmse_over_time: np.ndarray,
    target_rmse: float,
    time_steps: Optional[np.ndarray] = None
) -> float:
    """
    Compute time (or iterations) to reach target RMSE.

    Used for RQ2: How quickly does sensor calibration stabilize?

    Parameters
    ----------
    rmse_over_time : np.ndarray
        RMSE at each time step [T]
    target_rmse : float
        Target RMSE threshold
    time_steps : np.ndarray, optional
        Time values [T]. If None, uses indices.

    Returns
    -------
    float
        Time to stabilization (np.inf if never reached)
    """
    if time_steps is None:
        time_steps = np.arange(len(rmse_over_time))

    # Find first time RMSE drops below target
    idx = np.where(rmse_over_time <= target_rmse)[0]

    if len(idx) > 0:
        return float(time_steps[idx[0]])
    else:
        return float(np.inf)


class TransferEvaluator:
    """
    Comprehensive evaluator for transfer learning experiments.

    Computes all metrics relevant to RQ1 and RQ2.
    """

    def __init__(self, confidence: float = 0.95):
        """
        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level for interval metrics
        """
        self.confidence = confidence
        self.results = {}

    def evaluate_rq1(
        self,
        source_posterior_samples: np.ndarray,
        target_posterior_samples: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_values: np.ndarray
    ) -> dict:
        """
        Evaluate transfer quality for RQ1 (cross-regional generalization).

        Parameters
        ----------
        source_posterior_samples : np.ndarray
            Source posterior samples [N_s, D]
        target_posterior_samples : np.ndarray
            Target posterior samples [N_t, D]
        predictions : np.ndarray
            Predictions on target test set [N]
        uncertainties : np.ndarray
            Prediction uncertainties [N]
        true_values : np.ndarray
            True values [N]

        Returns
        -------
        dict
            Evaluation metrics
        """
        results = {
            'kl_divergence': kl_divergence_distributions(
                source_posterior_samples, target_posterior_samples
            ),
            **regression_metrics(predictions, true_values),
            'picp': prediction_interval_coverage_probability(
                predictions, uncertainties, true_values, self.confidence
            ),
            'mean_interval_width': mean_prediction_interval_width(
                uncertainties, self.confidence
            ),
            'calibration_error': calibration_error(
                predictions, uncertainties, true_values
            )
        }

        self.results['rq1'] = results
        return results

    def evaluate_rq2(
        self,
        rmse_over_time: np.ndarray,
        target_rmse: float,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_values: np.ndarray,
        time_steps: Optional[np.ndarray] = None
    ) -> dict:
        """
        Evaluate transfer quality for RQ2 (sensor adaptation).

        Parameters
        ----------
        rmse_over_time : np.ndarray
            RMSE at each time step [T]
        target_rmse : float
            Target RMSE for stabilization
        predictions : np.ndarray
            Final predictions [N]
        uncertainties : np.ndarray
            Final uncertainties [N]
        true_values : np.ndarray
            True values [N]
        time_steps : np.ndarray, optional
            Time values

        Returns
        -------
        dict
            Evaluation metrics
        """
        results = {
            'time_to_stabilization': time_to_stabilization(
                rmse_over_time, target_rmse, time_steps
            ),
            **regression_metrics(predictions, true_values),
            'picp': prediction_interval_coverage_probability(
                predictions, uncertainties, true_values, self.confidence
            ),
            'calibration_error': calibration_error(
                predictions, uncertainties, true_values
            )
        }

        self.results['rq2'] = results
        return results

    def print_summary(self):
        """Print formatted summary of results."""
        if 'rq1' in self.results:
            print("\n" + "="*50)
            print("RQ1: Cross-Regional Generalization")
            print("="*50)
            for key, value in self.results['rq1'].items():
                print(f"{key:25s}: {value:.4f}")

        if 'rq2' in self.results:
            print("\n" + "="*50)
            print("RQ2: Sensor Adaptation")
            print("="*50)
            for key, value in self.results['rq2'].items():
                print(f"{key:25s}: {value:.4f}")
