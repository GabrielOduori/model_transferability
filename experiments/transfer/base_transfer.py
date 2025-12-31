"""
Base Transfer Experiment Class
================================

Abstract base class for transfer learning experiments.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from pathlib import Path


class BaseTransferExperiment(ABC):
    """
    Abstract base class for transfer learning experiments.

    Provides common structure and utilities for running transfer learning
    experiments with different methods (Prior Tempering, OBTL, etc.).
    """

    def __init__(self, experiment_name: str, source_name: str = "Source domain"):
        """
        Initialize transfer experiment.

        Args:
            experiment_name: Name of the experiment
            source_name: Description of the source domain
        """
        self.experiment_name = experiment_name
        self.source_name = source_name
        self.results = []

    @abstractmethod
    def run_single_transfer(
        self,
        param_value: float,
        target_data: Dict,
        test_data: Dict
    ) -> Dict:
        """
        Run a single transfer experiment with given parameter value.

        Args:
            param_value: Transfer parameter (lambda for PT, delta for OBTL)
            target_data: Target domain training data
            test_data: Target domain test data

        Returns:
            Dict containing metrics for this parameter value
        """
        pass

    @abstractmethod
    def get_parameter_name(self) -> str:
        """Return the name of the transfer parameter (e.g., 'lambda', 'delta')."""
        pass

    def run(
        self,
        target_data: Dict,
        test_data: Dict,
        param_values: List[float]
    ) -> Dict:
        """
        Run transfer experiments for all parameter values.

        Args:
            target_data: Target domain training data
            test_data: Target domain test data
            param_values: List of parameter values to test

        Returns:
            Dict containing all experiment results
        """
        param_name = self.get_parameter_name()

        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"Source: {self.source_name}")
        print(f"Target: Synthetic Target domain data ({target_data['X'].shape[0]} samples)")
        print(f"{param_name} values: {param_values}")

        results = []

        for param_value in param_values:
            print(f"\n  {param_name} = {param_value:.2f}")

            # Run single transfer
            result = self.run_single_transfer(param_value, target_data, test_data)

            # Print metrics
            print(f"    RMSE: {result['rmse']:.2f} µg/m³")
            print(f"    MAE:  {result['mae']:.2f} µg/m³")
            print(f"    R²:   {result['r2']:.4f}")

            # Print additional metrics if present
            if 'weight_source' in result:
                print(f"    Transfer weights: Source={result['weight_source']:.3f}, "
                      f"Target={result['weight_target']:.3f}")

            results.append(result)

        # Package results
        self.results = results
        return {
            'experiment': self.experiment_name,
            'source': self.source_name,
            'target': 'Synthetic Target domain',
            'results': results,
            'best': min(results, key=lambda x: x['rmse'])
        }

    def get_best_result(self) -> Dict:
        """Get the result with the lowest RMSE."""
        if not self.results:
            raise ValueError("No results available. Run experiment first.")
        return min(self.results, key=lambda x: x['rmse'])

    def get_results_for_parameter(self, param_value: float) -> Dict:
        """Get results for a specific parameter value."""
        param_name = self.get_parameter_name()
        for result in self.results:
            if result[param_name] == param_value:
                return result
        raise ValueError(f"No results for {param_name}={param_value}")
