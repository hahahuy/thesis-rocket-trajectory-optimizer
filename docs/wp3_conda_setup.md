# WP3 Conda Environment Setup

## Environment Status

The **Thesis-rocket** conda environment has been verified and is ready for WP3 validation.

### Verified Packages

All required packages are installed:
- ✓ numpy 2.2.6 (NumPy 2.x - code is compatible)
- ✓ scipy 1.15.2
- ✓ matplotlib 3.10.7
- ✓ h5py 3.15.1
- ✓ casadi 3.7.2
- ✓ PyYAML 6.0.2
- ✓ torch 2.5.1
- ✓ pytest 8.4.2
- ✓ omegaconf 2.3.0

Optional:
- ○ hydra-core (not installed, but not required for WP3)

### NumPy 2.0 Compatibility

The codebase has been updated to be compatible with NumPy 2.x:
- Uses `np.bytes_` instead of deprecated `np.string_`
- Automatic fallback for NumPy < 2.0 environments

## Running Validation

### Option 1: Using Makefile (Recommended)

```bash
make -f Makefile.wp3 validate_wp3_conda
```

### Option 2: Direct Script

```bash
bash scripts/validate_wp3_conda.sh configs/dataset.yaml 12
```

### Option 3: Manual Steps

```bash
# Activate environment (if using conda activate)
conda activate Thesis-rocket

# Or use full path
export PYTHONPATH="$(pwd):$PYTHONPATH"
/home/hahuy/anaconda3/envs/Thesis-rocket/bin/python -m src.data.generator --config configs/dataset.yaml
/home/hahuy/anaconda3/envs/Thesis-rocket/bin/python -m src.data.preprocess --raw data/raw --out data/processed
/home/hahuy/anaconda3/envs/Thesis-rocket/bin/python -m src.eval.metrics --processed data/processed --raw data/raw --report reports/DATASET_CARD.json
/home/hahuy/anaconda3/envs/Thesis-rocket/bin/pytest -k "dataset or generator or preprocess" -v
```

## Check Requirements

To verify your environment has all required packages:

```bash
/home/hahuy/anaconda3/envs/Thesis-rocket/bin/python scripts/check_requirements.py
```

## Fixes Applied

1. **NumPy 2.0 Compatibility**: Updated `storage.py`, `preprocess.py`, and test files to use `np.bytes_` instead of `np.string_`
2. **Conda Script**: Created `validate_wp3_conda.sh` that uses the conda environment's Python
3. **Requirements Checker**: Added `check_requirements.py` to verify package installation

## Test Results

✅ All 30 tests passed in the conda environment
✅ Dataset generation successful
✅ Preprocessing successful
✅ Dataset card generated
✅ Visualization plots created

