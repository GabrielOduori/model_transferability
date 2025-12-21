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

# FusionGP transfer (optional - requires external package)
try:
    from .fusion_gp_transfer import (
        transfer_kernel_hyperparameters,
        transfer_inducing_points,
        transfer_likelihood_parameters,
        hybrid_fusion_transfer,
        TransferableFusionSVGP,
        get_transfer_summary as get_fusion_transfer_summary
    )
    _fusion_gp_available = True
except ImportError:
    _fusion_gp_available = False

__all__ = ['BaselineGP', 'SpatialTemporalGP', 'train_baseline_gp']

if _gam_ssm_available:
    __all__.extend([
        'transfer_lur_coefficients',
        'transfer_ssm_dynamics',
        'hybrid_transfer',
        'TransferableGAMSSM',
        'get_transfer_summary'
    ])

if _fusion_gp_available:
    __all__.extend([
        'transfer_kernel_hyperparameters',
        'transfer_inducing_points',
        'transfer_likelihood_parameters',
        'hybrid_fusion_transfer',
        'TransferableFusionSVGP',
        'get_fusion_transfer_summary'
    ])
