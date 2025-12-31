"""
Synthetic Data Generation
==========================

Generate synthetic target domain data for transfer learning experiments.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional


class SyntheticDataGenerator:
    """
    Generate synthetic target domain data with domain shift.

    This class handles generation, saving, and loading of synthetic
    spatiotemporal data for transfer learning experiments.
    """

    def __init__(self, seed: int = 42, base_path: Optional[Path] = None):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility
            base_path: Base path to project (auto-detected if None)
        """
        self.seed = seed
        self.base_path = base_path or Path(__file__).parent.parent.parent

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Data directory
        self.data_dir = self.base_path / 'data' / 'synthetic_target'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        n_target: int = 50,
        n_test: int = 100,
        n_features: int = 3,
        force_regenerate: bool = False
    ) -> Dict:
        """
        Generate or load synthetic target domain data.

        If saved data exists and matches parameters, loads it for reproducibility.
        Otherwise, generates new data and saves it.

        Args:
            n_target: Number of target training samples
            n_test: Number of target test samples
            n_features: Number of features (should match source model)
            force_regenerate: If True, regenerate even if saved file exists

        Returns:
            Dictionary containing:
                - target: {'X': tensor, 'y': tensor} - Training data
                - test: {'X': tensor, 'y': tensor} - Test data
                - metadata: Dict with generation parameters
        """
        data_file = self.data_dir / f'target_data_seed{self.seed}.npz'

        # Try to load existing data
        if data_file.exists() and not force_regenerate:
            loaded_data = self._load_saved_data(data_file, n_target, n_test)
            if loaded_data is not None:
                return loaded_data

        # Generate new data
        return self._generate_new_data(n_target, n_test, n_features)

    def _load_saved_data(
        self,
        data_file: Path,
        n_target: int,
        n_test: int
    ) -> Optional[Dict]:
        """Load saved data if metadata matches."""
        print(f"\n✓ Loading saved synthetic data: {data_file.name}")

        try:
            data = np.load(data_file)

            # Verify metadata matches
            if (int(data['n_target']) == n_target and
                int(data['n_test']) == n_test and
                int(data['seed']) == self.seed):

                print(f"  Metadata: n_target={data['n_target']}, "
                      f"n_test={data['n_test']}, seed={data['seed']}, "
                      f"domain_shift={data['domain_shift']}")

                # Convert to torch tensors
                X_target = torch.from_numpy(data['X_target'])
                y_target = torch.from_numpy(data['y_target'])
                X_test = torch.from_numpy(data['X_test'])
                y_test = torch.from_numpy(data['y_test'])

                print(f"  ✓ Using saved data for reproducibility")

                return {
                    'target': {'X': X_target, 'y': y_target},
                    'test': {'X': X_test, 'y': y_test},
                    'metadata': {
                        'seed': self.seed,
                        'n_target': n_target,
                        'n_test': n_test,
                        'domain_shift': float(data['domain_shift']),
                        'noise_std': float(data['noise_std']),
                        'saved_file': str(data_file),
                        'loaded_from_file': True
                    }
                }
            else:
                print(f"  ⚠️  Metadata mismatch, regenerating...")
                return None

        except Exception as e:
            print(f"  ⚠️  Error loading data: {e}, regenerating...")
            return None

    def _generate_new_data(
        self,
        n_target: int,
        n_test: int,
        n_features: int,
        domain_shift: float = 0.3,
        noise_std: float = 1.5
    ) -> Dict:
        """Generate new synthetic data."""
        print(f"\n🔄 Generating new synthetic target data (seed={self.seed})")
        print(f"  n_target={n_target}, n_test={n_test}, domain_shift={domain_shift}")

        # Reset seeds for consistency
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Generate target training data
        # Features: [x, y, time] - spatiotemporal coordinates (normalized)
        X_target = torch.randn(n_target, n_features) * 0.5 + 0.5

        # Add systematic offset for target domain (domain shift)
        X_target[:, :2] += domain_shift  # Spatial offset from source domain

        # Generate NO₂ concentrations with spatiotemporal pattern
        y_target = (
            15.0 +  # Base concentration
            5.0 * torch.sin(2 * np.pi * X_target[:, 0]) +  # Spatial pattern in x
            3.0 * torch.cos(2 * np.pi * X_target[:, 1]) +  # Spatial pattern in y
            2.0 * torch.sin(4 * np.pi * X_target[:, 2]) +  # Temporal pattern
            torch.randn(n_target) * noise_std  # Noise
        )

        # Generate test data (same distribution as target domain)
        X_test = torch.randn(n_test, n_features) * 0.5 + 0.5
        X_test[:, :2] += domain_shift  # Same spatial offset

        y_test = (
            15.0 +
            5.0 * torch.sin(2 * np.pi * X_test[:, 0]) +
            3.0 * torch.cos(2 * np.pi * X_test[:, 1]) +
            2.0 * torch.sin(4 * np.pi * X_test[:, 2]) +
            torch.randn(n_test) * noise_std
        )

        # Save data for reproducibility
        data_file = self.data_dir / f'target_data_seed{self.seed}.npz'
        np.savez(
            data_file,
            X_target=X_target.numpy(),
            y_target=y_target.numpy(),
            X_test=X_test.numpy(),
            y_test=y_test.numpy(),
            n_target=n_target,
            n_test=n_test,
            n_features=n_features,
            seed=self.seed,
            domain_shift=domain_shift,
            noise_std=noise_std
        )
        print(f"  ✓ Saved synthetic target data: {data_file.name}")

        return {
            'target': {'X': X_target, 'y': y_target},
            'test': {'X': X_test, 'y': y_test},
            'metadata': {
                'seed': self.seed,
                'n_target': n_target,
                'n_test': n_test,
                'domain_shift': domain_shift,
                'noise_std': noise_std,
                'saved_file': str(data_file),
                'loaded_from_file': False
            }
        }

    def get_data_path(self) -> Path:
        """Get path to data file for current seed."""
        return self.data_dir / f'target_data_seed{self.seed}.npz'

    def delete_saved_data(self):
        """Delete saved data file."""
        data_file = self.get_data_path()
        if data_file.exists():
            data_file.unlink()
            print(f"✓ Deleted: {data_file}")
