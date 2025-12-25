"""
Integration tests for complete transfer learning pipelines.
"""

import pytest
import torch
import numpy as np
from src.models.gp_model import BaselineGP, train_baseline_gp, predict_with_uncertainty
from src.transfer_methods.prior_tempering import TemperedGP, train_tempered_gp
from src.transfer_methods.obtl import OBTLGaussianProcess
from src.transfer_methods.dptr import DPTRGaussianProcess
from src.evaluation.metrics import regression_metrics, TransferEvaluator


class TestBaselineTransferPipeline:
    """Test baseline GP without transfer (for comparison)."""

    def test_baseline_pipeline(self, source_target_1d, device):
        """Test complete baseline pipeline."""
        # Ignore source data
        x_target, y_target = source_target_1d['target']
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Split target into train/test
        n_train = int(0.7 * len(x_target))
        x_train = x_target[:n_train]
        y_train = y_target[:n_train]
        x_test = x_target[n_train:]
        y_test = y_target[n_train:]

        # Train baseline GP
        likelihood = torch.nn.Module()
        import gpytorch
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BaselineGP(x_train, y_train, likelihood, ard=False)
        model = model.to(device)

        model, likelihood, losses = train_baseline_gp(
            model, likelihood, x_train, y_train, num_iter=100
        )

        # Predict on test set
        predictions, stds = predict_with_uncertainty(model, likelihood, x_test)

        # Evaluate
        predictions_np = predictions.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        metrics = regression_metrics(y_test_np, predictions_np)

        # Basic sanity checks
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert -1.0 <= metrics['r2'] <= 1.0
        assert np.isfinite(metrics['rmse'])


class TestPriorTemperingPipeline:
    """Test complete prior tempering transfer learning pipeline."""

    def test_full_prior_tempering_pipeline(self, source_target_1d, device):
        """Test end-to-end prior tempering transfer."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Step 1: Train source model
        import gpytorch
        likelihood_source = gpytorch.likelihoods.GaussianLikelihood()
        model_source = BaselineGP(x_source, y_source, likelihood_source, ard=False)
        model_source = model_source.to(device)

        model_source, likelihood_source, _ = train_baseline_gp(
            model_source, likelihood_source, x_source, y_source, num_iter=100
        )

        # Step 2: Extract source hyperparameters
        source_hyperparams = {
            'lengthscale': model_source.covar_module.base_kernel.lengthscale.detach(),
            'outputscale': model_source.covar_module.outputscale.detach(),
            'noise': likelihood_source.noise.detach()
        }

        # Step 3: Train target with tempering
        likelihood_target = gpytorch.likelihoods.GaussianLikelihood()
        model_target = TemperedGP(
            x_target, y_target, likelihood_target,
            source_hyperparams=source_hyperparams
        )
        model_target = model_target.to(device)

        model_target, likelihood_target, losses = train_tempered_gp(
            model_target, likelihood_target, x_target, y_target,
            beta=0.5, num_iter=100
        )

        # Step 4: Make predictions
        predictions, stds = predict_with_uncertainty(model_target, likelihood_target, x_target)

        # Step 5: Evaluate
        predictions_np = predictions.cpu().numpy()
        y_target_np = y_target.cpu().numpy()

        metrics = regression_metrics(y_target_np, predictions_np)

        assert np.isfinite(metrics['rmse'])
        assert np.isfinite(metrics['mae'])
        assert np.isfinite(metrics['r2'])

    def test_prior_tempering_vs_baseline(self, source_target_2d, device):
        """Compare prior tempering with baseline."""
        x_source, y_source = source_target_2d['source']
        x_target, y_target = source_target_2d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Split target for evaluation
        n_train = int(0.7 * len(x_target))
        x_train = x_target[:n_train]
        y_train = y_target[:n_train]
        x_test = x_target[n_train:]
        y_test = y_target[n_train:]

        import gpytorch

        # Train source
        likelihood_source = gpytorch.likelihoods.GaussianLikelihood()
        model_source = BaselineGP(x_source, y_source, likelihood_source, ard=True)
        model_source = model_source.to(device)
        model_source, likelihood_source, _ = train_baseline_gp(
            model_source, likelihood_source, x_source, y_source, num_iter=100
        )

        source_hyperparams = {
            'lengthscale': model_source.covar_module.base_kernel.lengthscale.detach(),
            'outputscale': model_source.covar_module.outputscale.detach(),
            'noise': likelihood_source.noise.detach()
        }

        # Baseline: train only on target
        likelihood_baseline = gpytorch.likelihoods.GaussianLikelihood()
        model_baseline = BaselineGP(x_train, y_train, likelihood_baseline, ard=True)
        model_baseline = model_baseline.to(device)
        model_baseline, likelihood_baseline, _ = train_baseline_gp(
            model_baseline, likelihood_baseline, x_train, y_train, num_iter=100
        )

        # Transfer: use prior tempering
        likelihood_transfer = gpytorch.likelihoods.GaussianLikelihood()
        model_transfer = TemperedGP(
            x_train, y_train, likelihood_transfer,
            source_hyperparams=source_hyperparams
        )
        model_transfer = model_transfer.to(device)
        model_transfer, likelihood_transfer, _ = train_tempered_gp(
            model_transfer, likelihood_transfer, x_train, y_train,
            beta=0.7, num_iter=100
        )

        # Evaluate both
        pred_baseline, _ = predict_with_uncertainty(model_baseline, likelihood_baseline, x_test)
        pred_transfer, _ = predict_with_uncertainty(model_transfer, likelihood_transfer, x_test)

        metrics_baseline = regression_metrics(y_test.cpu().numpy(), pred_baseline.cpu().numpy())
        metrics_transfer = regression_metrics(y_test.cpu().numpy(), pred_transfer.cpu().numpy())

        # Both should produce valid metrics
        assert np.isfinite(metrics_baseline['rmse'])
        assert np.isfinite(metrics_transfer['rmse'])


class TestOBTLPipeline:
    """Test complete OBTL transfer learning pipeline."""

    def test_full_obtl_pipeline(self, source_target_1d, device):
        """Test end-to-end OBTL transfer."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Initialize OBTL
        obtl = OBTLGaussianProcess(
            n_inducing_points=15,
            nu_0=10.0,
            delta=0.6
        )

        # Step 1: Fit source
        obtl.fit_source(x_source, y_source, num_iter=150)

        # Step 2: Transfer to target
        model_target, likelihood_target = obtl.transfer_to_target(
            x_target, y_target, num_iter=150
        )

        # Step 3: Make predictions
        model_target.eval()
        likelihood_target.eval()

        with torch.no_grad():
            output = model_target(x_target)
            predictions = likelihood_target(output).mean

        # Step 4: Evaluate
        predictions_np = predictions.cpu().numpy()
        y_target_np = y_target.cpu().numpy()

        metrics = regression_metrics(y_target_np, predictions_np)

        assert np.isfinite(metrics['rmse'])
        assert np.isfinite(metrics['mae'])

    def test_obtl_delta_comparison(self, source_target_2d, device):
        """Compare OBTL with different delta values."""
        x_source, y_source = source_target_2d['source']
        x_target, y_target = source_target_2d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        results = {}

        for delta in [0.0, 0.5, 1.0]:
            obtl = OBTLGaussianProcess(
                n_inducing_points=20,
                nu_0=10.0,
                delta=delta
            )

            obtl.fit_source(x_source, y_source, num_iter=100)
            model_target, likelihood_target = obtl.transfer_to_target(
                x_target, y_target, num_iter=100
            )

            model_target.eval()
            likelihood_target.eval()

            with torch.no_grad():
                output = model_target(x_target)
                predictions = likelihood_target(output).mean

            metrics = regression_metrics(
                y_target.cpu().numpy(),
                predictions.cpu().numpy()
            )

            results[delta] = metrics

        # All deltas should produce valid results
        for delta, metrics in results.items():
            assert np.isfinite(metrics['rmse'])


class TestDPTRPipeline:
    """Test complete DPTR transfer learning pipeline."""

    def test_full_dptr_pipeline(self, feature_mismatch_data, device):
        """Test end-to-end DPTR transfer."""
        x_source, y_source = feature_mismatch_data['source']
        x_target, y_target = feature_mismatch_data['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Initialize DPTR
        dptr = DPTRGaussianProcess(
            source_dim=x_source.shape[1],
            target_dim=x_target.shape[1],
            latent_dim=8,
            hidden_dim=32,
            beta=0.5
        )

        # Full transfer pipeline
        model_target, likelihood_target = dptr.fit_transfer(
            x_source, y_source,
            x_target, y_target,
            vae_epochs=50,
            gp_iterations=100,
            device=device
        )

        # Make predictions
        model_target.eval()
        likelihood_target.eval()
        dptr.vae.eval()

        z_target = dptr.vae.encode_target(x_target)

        with torch.no_grad():
            output = model_target(z_target)
            predictions = likelihood_target(output).mean

        # Evaluate
        predictions_np = predictions.cpu().numpy()
        y_target_np = y_target.cpu().numpy()

        metrics = regression_metrics(y_target_np, predictions_np)

        assert np.isfinite(metrics['rmse'])
        assert np.isfinite(metrics['mae'])

    def test_dptr_feature_alignment(self, feature_mismatch_data, device):
        """Test that DPTR aligns features from different dimensions."""
        x_source, y_source = feature_mismatch_data['source']
        x_target, y_target = feature_mismatch_data['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Different input dimensions
        assert x_source.shape[1] != x_target.shape[1]

        dptr = DPTRGaussianProcess(
            source_dim=x_source.shape[1],
            target_dim=x_target.shape[1],
            latent_dim=8
        )

        # Should handle feature mismatch
        model_target, likelihood_target = dptr.fit_transfer(
            x_source, y_source,
            x_target, y_target,
            vae_epochs=30,
            gp_iterations=50,
            device=device
        )

        # Both domains should be mapped to same latent dimension
        dptr.vae.eval()
        z_source = dptr.vae.encode_source(x_source)
        z_target = dptr.vae.encode_target(x_target)

        assert z_source.shape[1] == z_target.shape[1] == 8


class TestMultiMethodComparison:
    """Compare multiple transfer methods on same data."""

    def test_compare_all_methods(self, source_target_1d, device):
        """Compare baseline, prior tempering, and OBTL."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        import gpytorch

        results = {}

        # 1. Baseline (no transfer)
        likelihood_baseline = gpytorch.likelihoods.GaussianLikelihood()
        model_baseline = BaselineGP(x_target, y_target, likelihood_baseline, ard=False)
        model_baseline = model_baseline.to(device)
        model_baseline, likelihood_baseline, _ = train_baseline_gp(
            model_baseline, likelihood_baseline, x_target, y_target, num_iter=100
        )
        pred_baseline, std_baseline = predict_with_uncertainty(
            model_baseline, likelihood_baseline, x_target
        )
        results['baseline'] = regression_metrics(
            y_target.cpu().numpy(),
            pred_baseline.cpu().numpy()
        )

        # 2. Prior Tempering
        likelihood_source = gpytorch.likelihoods.GaussianLikelihood()
        model_source = BaselineGP(x_source, y_source, likelihood_source, ard=False)
        model_source = model_source.to(device)
        model_source, likelihood_source, _ = train_baseline_gp(
            model_source, likelihood_source, x_source, y_source, num_iter=100
        )

        source_hyperparams = {
            'lengthscale': model_source.covar_module.base_kernel.lengthscale.detach(),
            'outputscale': model_source.covar_module.outputscale.detach(),
            'noise': likelihood_source.noise.detach()
        }

        likelihood_tempering = gpytorch.likelihoods.GaussianLikelihood()
        model_tempering = TemperedGP(
            x_target, y_target, likelihood_tempering,
            source_hyperparams=source_hyperparams
        )
        model_tempering = model_tempering.to(device)
        model_tempering, likelihood_tempering, _ = train_tempered_gp(
            model_tempering, likelihood_tempering, x_target, y_target,
            beta=0.5, num_iter=100
        )
        pred_tempering, std_tempering = predict_with_uncertainty(
            model_tempering, likelihood_tempering, x_target
        )
        results['prior_tempering'] = regression_metrics(
            y_target.cpu().numpy(),
            pred_tempering.cpu().numpy()
        )

        # 3. OBTL
        obtl = OBTLGaussianProcess(n_inducing_points=15, delta=0.5)
        obtl.fit_source(x_source, y_source, num_iter=100)
        model_obtl, likelihood_obtl = obtl.transfer_to_target(
            x_target, y_target, num_iter=100
        )
        model_obtl.eval()
        likelihood_obtl.eval()
        with torch.no_grad():
            output_obtl = model_obtl(x_target)
            pred_obtl = likelihood_obtl(output_obtl).mean

        results['obtl'] = regression_metrics(
            y_target.cpu().numpy(),
            pred_obtl.cpu().numpy()
        )

        # All methods should produce valid results
        for method, metrics in results.items():
            assert np.isfinite(metrics['rmse']), f"{method} RMSE is not finite"
            assert np.isfinite(metrics['mae']), f"{method} MAE is not finite"

        # Print comparison
        print("\n=== Method Comparison ===")
        for method, metrics in results.items():
            print(f"{method:20s} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")


class TestEvaluatorIntegration:
    """Test TransferEvaluator with real transfer methods."""

    def test_evaluator_with_prior_tempering(self, source_target_1d, device):
        """Test evaluator with prior tempering results."""
        x_source, y_source = source_target_1d['source']
        x_target, y_target = source_target_1d['target']

        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        import gpytorch

        # Train baseline and transfer
        likelihood_baseline = gpytorch.likelihoods.GaussianLikelihood()
        model_baseline = BaselineGP(x_target, y_target, likelihood_baseline, ard=False)
        model_baseline = model_baseline.to(device)
        model_baseline, likelihood_baseline, _ = train_baseline_gp(
            model_baseline, likelihood_baseline, x_target, y_target, num_iter=100
        )

        # Train source
        likelihood_source = gpytorch.likelihoods.GaussianLikelihood()
        model_source = BaselineGP(x_source, y_source, likelihood_source, ard=False)
        model_source = model_source.to(device)
        model_source, likelihood_source, _ = train_baseline_gp(
            model_source, likelihood_source, x_source, y_source, num_iter=100
        )

        source_hyperparams = {
            'lengthscale': model_source.covar_module.base_kernel.lengthscale.detach(),
            'outputscale': model_source.covar_module.outputscale.detach(),
            'noise': likelihood_source.noise.detach()
        }

        likelihood_transfer = gpytorch.likelihoods.GaussianLikelihood()
        model_transfer = TemperedGP(
            x_target, y_target, likelihood_transfer,
            source_hyperparams=source_hyperparams
        )
        model_transfer = model_transfer.to(device)
        model_transfer, likelihood_transfer, _ = train_tempered_gp(
            model_transfer, likelihood_transfer, x_target, y_target,
            beta=0.5, num_iter=100
        )

        # Get predictions
        pred_baseline, std_baseline = predict_with_uncertainty(
            model_baseline, likelihood_baseline, x_target
        )
        pred_transfer, std_transfer = predict_with_uncertainty(
            model_transfer, likelihood_transfer, x_target
        )

        # Use evaluator
        evaluator = TransferEvaluator()

        # Get source and target predictions for KL divergence
        pred_source, _ = predict_with_uncertainty(
            model_source, likelihood_source, x_source
        )

        metrics = evaluator.evaluate_rq1(
            y_true=y_target.cpu().numpy(),
            y_pred_baseline=pred_baseline.cpu().numpy(),
            y_std_baseline=std_baseline.cpu().numpy(),
            y_pred_transfer=pred_transfer.cpu().numpy(),
            y_std_transfer=std_transfer.cpu().numpy(),
            source_predictions=pred_source.cpu().numpy(),
            target_predictions=pred_transfer.cpu().numpy()
        )

        # All metrics should be finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite"

        evaluator.print_summary(metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
