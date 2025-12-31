"""
Visualization module for transfer learning experiments.

This module contains functions to create publication-quality visualizations
for Bayesian transfer learning methods, including:

- Conceptual diagrams (prior tempering, OBTL mechanisms)
- Spatial comparison maps (standard vs probabilistic transfer)
- Performance metrics plots (RMSE, R², MAE vs hyperparameters)
- Training diagnostics (loss convergence, gain/loss analysis, weight evolution)
"""

from .conceptual_diagrams import plot_prior_tempering_concept
from .spatial_maps import plot_spatial_comparison, generate_example_spatial_data
from .training_diagnostics import (
    plot_loss_convergence,
    plot_transfer_gain_loss,
    plot_transfer_weight_evolution
)

__all__ = [
    'plot_prior_tempering_concept',
    'plot_spatial_comparison',
    'generate_example_spatial_data',
    'plot_loss_convergence',
    'plot_transfer_gain_loss',
    'plot_transfer_weight_evolution'
]
