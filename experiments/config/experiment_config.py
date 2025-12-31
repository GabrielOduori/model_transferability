"""
Experiment Configuration
========================

Centralized configuration for transfer learning experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for transfer learning experiments."""

    # ========== Data Generation ==========
    n_target: int = 50
    """Number of target domain training samples"""

    n_test: int = 100
    """Number of target domain test samples"""

    n_features: int = 3
    """Number of features (should match source models)"""

    seed: int = 42
    """Random seed for reproducibility"""

    force_regenerate: bool = False
    """Force regeneration of synthetic data even if saved version exists"""

    # ========== Transfer Parameters ==========
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0])
    """Temperature parameters for Prior Tempering"""

    delta_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])
    """Transfer strength parameters for OBTL"""

    n_inducing_points: int = 30
    """Number of inducing points for OBTL"""

    nu_0: float = 35.0
    """Degrees of freedom for OBTL source prior (must be > d+1 where d=n_inducing_points)"""

    # ========== Model Paths ==========
    base_path: Optional[Path] = None
    """Base path to project root (auto-detected if None)"""

    fusiongp_path: Optional[Path] = None
    """Path to FusionGP model file"""

    gam_path: Optional[Path] = None
    """Path to GAM model file"""

    ssm_path: Optional[Path] = None
    """Path to SSM model file"""

    gam_data_path: Optional[Path] = None
    """Path to GAM training data"""

    # ========== Output Configuration ==========
    results_dir: Optional[Path] = None
    """Directory for saving results"""

    save_figures: bool = True
    """Generate and save visualization figures"""

    save_tables: bool = True
    """Generate and save CSV/LaTeX tables"""

    save_json: bool = True
    """Save results as JSON"""

    # ========== Training Configuration ==========
    num_iter_pt: int = 100
    """Number of training iterations for Prior Tempering"""

    num_iter_obtl_source: int = 100
    """Number of training iterations for OBTL source fit"""

    num_iter_obtl_target: int = 200
    """Number of training iterations for OBTL target transfer"""

    verbose: bool = True
    """Print training progress"""

    def __post_init__(self):
        """Set up derived paths after initialization."""

        # Auto-detect base path
        if self.base_path is None:
            # Assumes config is in experiments/config/
            self.base_path = Path(__file__).parent.parent.parent

        # Set default model paths
        if self.fusiongp_path is None:
            self.fusiongp_path = (
                self.base_path / 'models' / 'fusiongp' / 'dublin' / 'fusiongp_model.pth'
            )

        if self.gam_path is None:
            self.gam_path = (
                self.base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'gam.pkl'
            )

        if self.ssm_path is None:
            self.ssm_path = (
                self.base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'ssm.pkl'
            )

        if self.gam_data_path is None:
            self.gam_data_path = (
                self.base_path / 'models' / 'gam_ssm_lur' / 'dublin' / 'training_data.npz'
            )

        # Set default results directory
        if self.results_dir is None:
            self.results_dir = self.base_path / 'results'

        # Ensure paths are Path objects
        self.base_path = Path(self.base_path)
        self.fusiongp_path = Path(self.fusiongp_path)
        self.gam_path = Path(self.gam_path)
        self.ssm_path = Path(self.ssm_path)
        self.gam_data_path = Path(self.gam_data_path)
        self.results_dir = Path(self.results_dir)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'data': {
                'n_target': self.n_target,
                'n_test': self.n_test,
                'n_features': self.n_features,
                'seed': self.seed,
            },
            'transfer': {
                'prior_tempering': {
                    'lambda_values': self.lambda_values,
                    'num_iter': self.num_iter_pt,
                },
                'obtl': {
                    'delta_values': self.delta_values,
                    'num_iter_source': self.num_iter_obtl_source,
                    'num_iter_target': self.num_iter_obtl_target,
                }
            },
            'models': {
                'fusiongp': str(self.fusiongp_path),
                'gam': str(self.gam_path),
                'ssm': str(self.ssm_path),
                'gam_data': str(self.gam_data_path),
            },
            'output': {
                'results_dir': str(self.results_dir),
                'save_figures': self.save_figures,
                'save_tables': self.save_tables,
                'save_json': self.save_json,
            }
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        # Flatten nested dictionary
        flat_config = {}

        if 'data' in config_dict:
            flat_config.update(config_dict['data'])

        if 'transfer' in config_dict:
            if 'prior_tempering' in config_dict['transfer']:
                pt_config = config_dict['transfer']['prior_tempering']
                flat_config['lambda_values'] = pt_config.get('lambda_values')
                flat_config['num_iter_pt'] = pt_config.get('num_iter', 100)

            if 'obtl' in config_dict['transfer']:
                obtl_config = config_dict['transfer']['obtl']
                flat_config['delta_values'] = obtl_config.get('delta_values')
                flat_config['num_iter_obtl_source'] = obtl_config.get('num_iter_source', 100)
                flat_config['num_iter_obtl_target'] = obtl_config.get('num_iter_target', 200)

        if 'models' in config_dict:
            flat_config['fusiongp_path'] = config_dict['models'].get('fusiongp')
            flat_config['gam_path'] = config_dict['models'].get('gam')
            flat_config['ssm_path'] = config_dict['models'].get('ssm')
            flat_config['gam_data_path'] = config_dict['models'].get('gam_data')

        if 'output' in config_dict:
            output_config = config_dict['output']
            flat_config['results_dir'] = output_config.get('results_dir')
            flat_config['save_figures'] = output_config.get('save_figures', True)
            flat_config['save_tables'] = output_config.get('save_tables', True)
            flat_config['save_json'] = output_config.get('save_json', True)

        return cls(**{k: v for k, v in flat_config.items() if v is not None})

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"ExperimentConfig(\n"
            f"  Data: n_target={self.n_target}, n_test={self.n_test}, seed={self.seed}\n"
            f"  PT: λ={self.lambda_values}\n"
            f"  OBTL: δ={self.delta_values}\n"
            f"  Output: {self.results_dir}\n"
            f")"
        )


# Predefined configurations
DEFAULT_CONFIG = ExperimentConfig()

QUICK_TEST_CONFIG = ExperimentConfig(
    n_target=20,
    n_test=30,
    lambda_values=[0.5, 1.0],
    delta_values=[0.5, 1.0],
    num_iter_pt=50,
    num_iter_obtl_source=50,
    num_iter_obtl_target=100,
)

FULL_SWEEP_CONFIG = ExperimentConfig(
    lambda_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    delta_values=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
)
