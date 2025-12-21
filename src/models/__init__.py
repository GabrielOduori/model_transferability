from .gp_model import BaselineGP, SpatialTemporalGP, train_baseline_gp

# GAM-SSM-LUR transfer (optional - requires external package)
try:
    from .gal_ssm_lur import (
        transfer_lur_coefficients,
        transfer_ssm_dynamics,
        hybrid_transfer,
        TransferableGAMSSM,
        get_transfer_summary
    )
    _gam_ssm_available = True
except ImportError:
    _gam_ssm_available = False

__all__ = ['BaselineGP', 'SpatialTemporalGP', 'train_baseline_gp']

if _gam_ssm_available:
    __all__.extend([
        'transfer_lur_coefficients',
        'transfer_ssm_dynamics',
        'hybrid_transfer',
        'TransferableGAMSSM',
        'get_transfer_summary'
    ])
