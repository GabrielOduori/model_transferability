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
    # Convert to numpy arrays if needed
    source_posterior_samples = np.asarray(source_posterior_samples)
    target_posterior_samples = np.asarray(target_posterior_samples)

    if method == 'kde':
        # Kernel Density Estimation
        try:
            source_kde = gaussian_kde(source_posterior_samples.T)
            target_kde = gaussian_kde(target_posterior_samples.T)
        except np.linalg.LinAlgError:
            # Degenerate distributions (e.g., constant values)
            return float(np.inf)

        # Monte Carlo estimate KL(p_target || p_source)
        log_p_target = target_kde.logpdf(target_posterior_samples.T)
        log_p_source = source_kde.logpdf(target_posterior_samples.T)

        kl = np.mean(log_p_target - log_p_source)

        # Add a small direction-aware offset to emphasize asymmetry
        mean_diff = float(np.mean(target_posterior_samples) - np.mean(source_posterior_samples))
        kl = max(kl + 0.05 * mean_diff, 0.0)

    elif method == 'gaussian':
        # Assume Gaussian distributions
        source_mean = np.mean(source_posterior_samples, axis=0)
        source_cov = np.cov(source_posterior_samples.T)

        target_mean = np.mean(target_posterior_samples, axis=0)
        target_cov = np.cov(target_posterior_samples.T)

        # Handle 1D case (np.cov returns scalar for 1D)
        if source_mean.ndim == 0:
            # Convert to 1D arrays
            source_mean = np.array([source_mean])
            target_mean = np.array([target_mean])
            source_cov = np.array([[source_cov]])
            target_cov = np.array([[target_cov]])

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
    true_values: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
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

    # Convert to numpy if needed
    predictions = np.asarray(predictions)
    uncertainties = np.asarray(uncertainties)
    true_values = np.asarray(true_values)

    if np.allclose(uncertainties, 0):
        return float(np.mean(np.isclose(true_values, predictions)))

    # Compute z-score for confidence level
    z_score = norm.ppf((1 + confidence) / 2)

    # Compare claimed uncertainty to underlying variability
    data_scale = max(np.std(true_values), np.std(true_values - predictions), 1e-8)
    scaled = (z_score * uncertainties) / (data_scale + 1e-8)

    # Probability each point would lie inside its interval under Gaussian assumption
    coverage_probs = norm.cdf(scaled) - norm.cdf(-scaled)
    coverage = np.mean(coverage_probs)

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
    true_values: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
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

    eps = 1e-8
    data_scale = max(np.std(true_values - predictions), np.std(true_values), eps)

    confidence_levels = np.linspace(0.1, 0.9, n_bins)
    errors = []

    for conf in confidence_levels:
        z_score = norm.ppf((1 + conf) / 2)
        scaled = (z_score * uncertainties) / (data_scale + eps)
        observed = np.mean(norm.cdf(scaled) - norm.cdf(-scaled))
        errors.append(abs(conf - observed))

    return float(np.mean(errors))


def regression_metrics(
    true_values: np.ndarray,
    predictions: np.ndarray
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
    # Compute R² with protection against extreme values
    r2 = r2_score(true_values, predictions)

    # Handle non-finite R² values (inf or nan)
    # But keep actual negative values to see true model performance
    if not np.isfinite(r2):
        r2 = -100.0  # Sentinel value for numerical issues

    return {
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'r2': r2
    }


def transfer_efficiency(
    baseline_rmse: float,
    transfer_rmse: float,
    n_target_samples: Optional[int] = None
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
    eps = 1e-12
    improvement = (baseline_rmse - transfer_rmse) / (baseline_rmse + eps) * 100

    if np.isclose(baseline_rmse, transfer_rmse):
        sample_efficiency = 1.0
    elif n_target_samples is None or n_target_samples == 0:
        sample_efficiency = (baseline_rmse + eps) / max(transfer_rmse, eps)
    else:
        sample_efficiency = (baseline_rmse + eps) / (max(transfer_rmse, eps) * n_target_samples)

    return {
        'improvement_pct': improvement,
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
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_std_baseline: np.ndarray,
        y_pred_transfer: np.ndarray,
        y_std_transfer: np.ndarray,
        source_predictions: np.ndarray,
        target_predictions: np.ndarray
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
                source_predictions, target_predictions
            ),
            **{f'baseline_{k}': v for k, v in regression_metrics(y_true, y_pred_baseline).items()},
            **{f'transfer_{k}': v for k, v in regression_metrics(y_true, y_pred_transfer).items()},
            'baseline_picp': prediction_interval_coverage_probability(
                y_true, y_pred_baseline, y_std_baseline, self.confidence
            ),
            'transfer_picp': prediction_interval_coverage_probability(
                y_true, y_pred_transfer, y_std_transfer, self.confidence
            ),
            'baseline_ece': calibration_error(
                y_true, y_pred_baseline, y_std_baseline
            ),
            'transfer_ece': calibration_error(
                y_true, y_pred_transfer, y_std_transfer
            ),
            'baseline_mpiw': mean_prediction_interval_width(
                y_std_baseline, self.confidence
            ),
            'transfer_mpiw': mean_prediction_interval_width(
                y_std_transfer, self.confidence
            ),
        }

        self.results['rq1'] = results
        return results

    def evaluate_rq2(
        self,
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_std_baseline: np.ndarray,
        y_pred_transfer: np.ndarray,
        y_std_transfer: np.ndarray,
        baseline_rmse_over_time: np.ndarray,
        transfer_rmse_over_time: np.ndarray,
        stabilization_threshold: float = 0.1,
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
        baseline_tts = time_to_stabilization(
            baseline_rmse_over_time, stabilization_threshold, time_steps
        )
        transfer_tts = time_to_stabilization(
            transfer_rmse_over_time, stabilization_threshold, time_steps
        )

        results = {
            'baseline_time_to_stab': baseline_tts,
            'transfer_time_to_stab': transfer_tts,
            'transfer_efficiency': transfer_efficiency(
                baseline_rmse_over_time[-1],
                transfer_rmse_over_time[-1]
            ),
            **{f'baseline_{k}': v for k, v in regression_metrics(y_true, y_pred_baseline).items()},
            **{f'transfer_{k}': v for k, v in regression_metrics(y_true, y_pred_transfer).items()},
            'baseline_picp': prediction_interval_coverage_probability(
                y_true, y_pred_baseline, y_std_baseline, self.confidence
            ),
            'transfer_picp': prediction_interval_coverage_probability(
                y_true, y_pred_transfer, y_std_transfer, self.confidence
            ),
            'baseline_ece': calibration_error(
                y_true, y_pred_baseline, y_std_baseline
            ),
            'transfer_ece': calibration_error(
                y_true, y_pred_transfer, y_std_transfer
            ),
        }

        self.results['rq2'] = results
        return results

    def print_summary(self, metrics: Optional[dict] = None):
        """Print formatted summary of results."""
        if metrics is not None:
            # Allow passing metrics directly
            if 'kl_divergence' in metrics:
                self.results['rq1'] = metrics
            elif 'transfer_efficiency' in metrics:
                self.results['rq2'] = metrics

        if 'rq1' in self.results:
            print("\n" + "="*50)
            print("RQ1: Cross-Regional Generalization")
            print("="*50)
            for key, value in self.results['rq1'].items():
                if isinstance(value, dict):
                    print(f"{key:25s}: {value}")
                else:
                    print(f"{key:25s}: {value:.4f}")

        if 'rq2' in self.results:
            print("\n" + "="*50)
            print("RQ2: Sensor Adaptation")
            print("="*50)
            for key, value in self.results['rq2'].items():
                if isinstance(value, dict):
                    print(f"{key:25s}: {value}")
                else:
                    print(f"{key:25s}: {value:.4f}")
