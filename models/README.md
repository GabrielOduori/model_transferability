# Trained Models Directory

This directory contains pre-trained models for transfer learning experiments.

## Structure

```
models/
├── fusiongp/
│   ├── dublin/
│   │   ├── fusiongp_dublin_trained.pt      # Trained Dublin FusionGP model
│   │   ├── fusiongp_dublin_data.pkl        # Dublin training data (X, y)
│   │   └── fusiongp_dublin_metadata.json   # Model config and metrics
│   └── cork/
│       └── (Cork-specific models if any)
│
├── gam_ssm_lur/
│   ├── dublin/
│   │   ├── gam_ssm_lur_dublin_trained.pt   # Trained Dublin GAM-SSM-LUR
│   │   ├── gam_ssm_lur_dublin_data.pkl     # Dublin training data
│   │   └── gam_ssm_lur_dublin_metadata.json
│   └── cork/
│       └── (Cork-specific models if any)
│
└── README.md (this file)
```

## File Formats

### Model Checkpoints (.pt)

PyTorch checkpoint dictionary containing:
```python
{
    'model_state_dict': OrderedDict,      # Model parameters
    'likelihood_state_dict': OrderedDict, # Likelihood parameters (if applicable)
    'optimizer_state_dict': dict,         # Optimizer state (for resuming training)
    'config': dict,                       # Model configuration
    'history': dict,                      # Training history (losses, metrics)
    'epoch': int,                         # Last epoch
    'best_metric': float                  # Best validation metric
}
```

### Training Data (.pkl)

Pickle file containing:
```python
{
    'X_train': np.ndarray or torch.Tensor,  # Training features
    'y_train': np.ndarray or torch.Tensor,  # Training targets
    'X_val': np.ndarray or torch.Tensor,    # Validation features (optional)
    'y_val': np.ndarray or torch.Tensor,    # Validation targets (optional)
    'feature_names': list,                  # Feature column names
    'timestamps': np.ndarray,               # Temporal information (if applicable)
    'coordinates': np.ndarray,              # Spatial coordinates (if applicable)
    'metadata': dict                        # Additional metadata
}
```

### Metadata (.json)

Model configuration and performance:
```json
{
    "model_type": "FusionGP",
    "location": "Dublin",
    "training_date": "2025-12-25",
    "n_samples": 1000,
    "n_features": 10,
    "performance": {
        "train_rmse": 2.5,
        "val_rmse": 3.1,
        "test_rmse": 3.0,
        "train_r2": 0.95,
        "val_r2": 0.92
    },
    "hyperparameters": {
        "n_inducing": 200,
        "kernel_type": "matern32",
        "learning_rate": 0.01,
        "n_epochs": 100
    }
}
```

## Usage

### Loading Models

```python
from pathlib import Path
import torch
import pickle

# Load FusionGP
fusiongp_checkpoint = torch.load('models/fusiongp/dublin/fusiongp_dublin_trained.pt')
fusiongp_model = FusionGP(...)
fusiongp_model.load_state_dict(fusiongp_checkpoint['model_state_dict'])

# Load training data
with open('models/fusiongp/dublin/fusiongp_dublin_data.pkl', 'rb') as f:
    dublin_data = pickle.load(f)

X_train = dublin_data['X_train']
y_train = dublin_data['y_train']
```

### Saving New Models

```python
import torch
import pickle
import json
from datetime import datetime

# Save model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'likelihood_state_dict': likelihood.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config,
    'history': training_history,
    'epoch': final_epoch,
    'best_metric': best_val_rmse
}
torch.save(checkpoint, 'models/fusiongp/dublin/fusiongp_dublin_trained.pt')

# Save training data
data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'feature_names': feature_names,
    'timestamps': timestamps,
    'coordinates': coords,
    'metadata': {'description': 'Dublin NO2 data 2020-2023'}
}
with open('models/fusiongp/dublin/fusiongp_dublin_data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Save metadata
metadata = {
    'model_type': 'FusionGP',
    'location': 'Dublin',
    'training_date': datetime.now().isoformat(),
    'n_samples': len(X_train),
    'n_features': X_train.shape[1],
    'performance': {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse
    },
    'hyperparameters': model_config
}
with open('models/fusiongp/dublin/fusiongp_dublin_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Transfer Learning Workflow

1. **Train source models** (Dublin):
   ```bash
   # Train FusionGP on Dublin data
   python experiments/train_fusiongp_dublin.py
   # Saves to models/fusiongp/dublin/

   # Train GAM-SSM-LUR on Dublin data
   python experiments/train_gam_ssm_lur_dublin.py
   # Saves to models/gam_ssm_lur/dublin/
   ```

2. **Transfer to target** (Cork):
   ```bash
   # Load Dublin models and transfer to Cork
   python experiments/run_model_transfer_experiments.py
   # Uses models from models/{fusiongp,gam_ssm_lur}/dublin/
   # Generates Cork predictions
   ```

## Notes

- `.pt` files are included in `.gitignore` due to size
- Models can be stored on external storage or downloaded separately
- For reproducibility, always include metadata files in git
- Large data files (>100MB) should use Git LFS or external storage

## Current Status

- [ ] FusionGP Dublin model copied/saved
- [ ] FusionGP Dublin training data extracted
- [ ] GAM-SSM-LUR Dublin model copied/saved
- [ ] GAM-SSM-LUR Dublin training data extracted

---

**Last Updated**: 2025-12-25
