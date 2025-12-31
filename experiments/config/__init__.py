"""Configuration management for transfer learning experiments."""

from .experiment_config import (
    ExperimentConfig,
    DEFAULT_CONFIG,
    QUICK_TEST_CONFIG,
    FULL_SWEEP_CONFIG
)

__all__ = [
    'ExperimentConfig',
    'DEFAULT_CONFIG',
    'QUICK_TEST_CONFIG',
    'FULL_SWEEP_CONFIG'
]
