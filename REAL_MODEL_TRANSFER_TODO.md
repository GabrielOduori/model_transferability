# Real Model Transfer Learning - TODO

**Goal**: Complete the real model transfer learning experiments for thesis chapter

**Current Status**: Framework ready, needs model loading implementation

---

## ✅ Completed

1. **Transfer learning methods implemented**:
   - ✅ OBTL (Optimal Bayesian Transfer Learning)
   - ✅ Prior Tempering
   - ✅ DPTR (Deep Probabilistic Transfer Regression)

2. **Synthetic experiments working**:
   - ✅ `run_transfer_experiments.py` - 2×2 framework with synthetic data
   - ✅ `run_full_transfer_experiments.py` - OBTL vs Prior Tempering comparison
   - ✅ All demos (OBTL, Prior Tempering, DPTR) running successfully

3. **Infrastructure**:
   - ✅ Test suite (66/108 passing - 61%)
   - ✅ 9 commits completed and organized
   - ✅ Clean git state

---

## 🔨 TODO: Complete Real Model Transfer

### Task 1: Locate GAM-SSM-LUR Model

**What we need**: Path to trained Dublin GAM-SSM-LUR model checkpoint

**Current knowledge**:
- ✅ Found FusionGP model: `gam_ssm_lur/fusionGP2/fusiongp/notebooks/fusiongp_model.pt`
- ❓ GAM-SSM-LUR model: **Location unknown**

**Action items**:
```bash
# Search for GAM-SSM-LUR saved models
find gam_ssm_lur/src -name "*.pt" -o -name "*.pkl" -o -name "*.pickle"

# Check for saved experiments
ls -la gam_ssm_lur/experiments/
ls -la gam_ssm_lur/checkpoints/ 2>/dev/null
```

**Questions to answer**:
1. Where is the trained GAM-SSM-LUR model saved?
2. What format is it in? (.pt, .pkl, custom?)
3. Does it include the full model or just parameters?

---

### Task 2: Locate Dublin Training Data

**What we need**: Dublin training data (X, y) for OBTL covariance extraction

**Why needed**: OBTL requires source domain data to compute source covariance structure

**Possible locations**:
```bash
# Check for data files
ls -la data/
ls -la gam_ssm_lur/data/
ls -la src/data/

# Look for saved datasets
find . -name "*dublin*.csv" -o -name "*dublin*.pkl" -o -name "*dublin*.h5"
```

**Questions to answer**:
1. Where is Dublin training data saved?
2. What format? (CSV, pickle, HDF5, NetCDF?)
3. What features are included?

---

### Task 3: Complete FusionGP Model Loading

**File**: `experiments/run_real_model_transfer.py`

**Current status**: Checkpoint structure identified, loading function needs completion

**What we know**:
```python
# Checkpoint structure:
{
  'model_state_dict': OrderedDict(...),
  'likelihood_state_dict': OrderedDict(...),
  'optimizer_state_dict': dict,
  'history': dict,
  'config': dict
}

# Model architecture (from state_dict):
- variational_strategy.inducing_points: [200, 3]
- Spatial kernel (2D lengthscales)
- Temporal kernel (1D lengthscale)
- Product kernel
```

**Implementation options**:

**Option A: Load Full FusionSVGP** (if you can access FusionGP repo classes):
```python
from models.svgp import FusionSVGP
from models.likelihoods import MultiSourceLikelihood

model = FusionSVGP(n_inducing=200)
model.load_state_dict(checkpoint['model_state_dict'])

likelihood = MultiSourceLikelihood()
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
```

**Option B: Extract to BaselineGP** (simpler, works with our transfer methods):
```python
from src.models.gp_model import BaselineGP

# Use inducing points as pseudo-training data
train_x = checkpoint['model_state_dict']['variational_strategy.inducing_points'][:10, :]
train_y = checkpoint['model_state_dict']['variational_strategy._variational_distribution.variational_mean'][:10]

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = BaselineGP(train_x, train_y, likelihood)

# Load compatible hyperparameters
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

**Action**: Choose Option A or B and implement in `load_dublin_fusiongp()`

---

### Task 4: Implement GAM-SSM-LUR Loading

**File**: `experiments/run_real_model_transfer.py`

**Function**: `load_dublin_gam_ssm_lur(model_path)`

**Template**:
```python
def load_dublin_gam_ssm_lur(model_path: str):
    """Load pre-trained Dublin GAM-SSM-LUR model."""

    # Import from gam_ssm_lur repo
    from gam_ssm_lur.models.hybrid import HybridGAMSSM  # or whatever the class is

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Reconstruct model
    model = HybridGAMSSM(...)  # with appropriate parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model
```

**Action**:
1. Find the GAM-SSM-LUR model class in `gam_ssm_lur/src/gam_ssm_lur/models/`
2. Determine initialization parameters
3. Implement loading function

---

### Task 5: Adapt Transfer Methods for Real Models

**Challenge**: FusionSVGP and GAM-SSM-LUR have different APIs than BaselineGP

**Solutions**:

**For Prior Tempering**:
- Current: Works with any GP that has `covar_module`, `mean_module`, etc.
- ✅ Should work directly if we extract to BaselineGP (Option B)
- ⚠️  May need wrapper if using full FusionSVGP (Option A)

**For OBTL**:
- Requires source training data (X_source, y_source)
- Extracts covariance at inducing points
- Should work once we have Dublin data

**Action**:
1. Test Prior Tempering with loaded FusionGP
2. Prepare Dublin data for OBTL
3. Create wrapper functions if needed

---

### Task 6: Generate Synthetic Cork Data

**File**: `experiments/run_real_model_transfer.py`

**Function**: `generate_synthetic_cork_data()` ✅ Already implemented

**What it does**:
- Creates 50 Cork training samples
- Creates 100 Cork test samples
- Simulates domain shift from Dublin (offset + different variance)

**Action**: ✅ Done - verify parameters are reasonable

---

### Task 7: Run Transfer Experiments

**Once Tasks 1-5 complete**:

```bash
# Run real model transfer experiments
python experiments/run_real_model_transfer.py
```

**Expected output**:
```
📦 Loading Dublin FusionGP model
   ✓ FusionGP loaded successfully

📦 Generating synthetic Cork data
   Target: 50 samples
   Test: 100 samples

EXPERIMENT: FusionGP Transfer with Prior Tempering
  β = 0.3
    RMSE: XX.XX µg/m³
    MAE:  XX.XX µg/m³
    R²:   0.XXXX
  ...

✓ Results saved to: results/real_model_transfer/real_transfer_YYYYMMDD_HHMMSS.json
```

---

## 📋 Quick Action Checklist

- [ ] **Find GAM-SSM-LUR model checkpoint**
- [ ] **Find Dublin training data (X, y)**
- [ ] **Complete `load_dublin_fusiongp()` implementation**
- [ ] **Complete `load_dublin_gam_ssm_lur()` implementation**
- [ ] **Test FusionGP loading**
- [ ] **Test GAM-SSM-LUR loading**
- [ ] **Run Prior Tempering transfer with FusionGP**
- [ ] **Run OBTL transfer with FusionGP (needs Dublin data)**
- [ ] **Run transfers with GAM-SSM-LUR**
- [ ] **Generate thesis statistics and figures**

---

## 🎯 Expected Thesis Results

Once complete, you'll have:

1. **Real Dublin models → Synthetic Cork transfer**
   - FusionGP with Prior Tempering
   - FusionGP with OBTL
   - GAM-SSM-LUR with Prior Tempering
   - GAM-SSM-LUR with OBTL

2. **Performance metrics**:
   - RMSE, MAE, R² for each method
   - Improvement over no-transfer baseline
   - Best β/δ parameters

3. **Thesis-ready outputs**:
   - JSON results files
   - CSV tables
   - Performance comparison plots

---

## 📝 Notes

**Current experiment files**:
- `run_transfer_experiments.py` - Synthetic source & target (working ✅)
- `run_full_transfer_experiments.py` - OBTL vs Prior Tempering comparison (working ✅)
- `run_real_model_transfer.py` - Real models → Synthetic target (needs completion ⚠️)

**Next session focus**: Complete Tasks 1-5 to enable real model transfer experiments

---

**Created**: 2025-12-25
**Status**: Ready for implementation
**Priority**: High (core thesis experiment)
