#!/usr/bin/env bash
# WP3 pre-flight validation script (Conda environment version)
set -euo pipefail

# Detect conda environment
CONDA_ENV="${CONDA_ENV:-Thesis-rocket}"
CONDA_PYTHON="/home/hahuy/anaconda3/envs/${CONDA_ENV}/bin/python"
CONDA_PYTEST="/home/hahuy/anaconda3/envs/${CONDA_ENV}/bin/pytest"

# Check if conda environment exists
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: Conda environment '$CONDA_ENV' not found at $CONDA_PYTHON"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "=== WP3 Pre-Flight Validation (Conda: $CONDA_ENV) ==="
echo "Using Python: $CONDA_PYTHON"

# Check required packages
echo ""
echo "Checking required packages..."
$CONDA_PYTHON -c "
import sys
missing = []
try:
    import numpy
    import scipy
    import h5py
    import matplotlib
    import casadi
    import yaml
    import torch
    print('✓ All core packages available')
except ImportError as e:
    print(f'✗ Missing package: {e.name}')
    missing.append(e.name)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Missing required packages. Install with:"
    echo "  conda activate $CONDA_ENV"
    echo "  pip install numpy scipy h5py matplotlib casadi pyyaml torch"
    exit 1
fi

# Configuration
CONFIG="${1:-configs/dataset.yaml}"
SMOKE_N="${2:-12}"  # Small smoke test size

echo ""
echo "Using config: $CONFIG"
echo "Smoke test size: $SMOKE_N"

# Create smoke config
SMOKE_CONFIG="configs/dataset_smoke.yaml"
$CONDA_PYTHON <<EOF
import yaml
with open("$CONFIG", "r") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["n_train"] = $SMOKE_N // 2
cfg["dataset"]["n_val"] = $SMOKE_N // 4
cfg["dataset"]["n_test"] = $SMOKE_N // 4
cfg["dataset"]["parallel_workers"] = 2
with open("$SMOKE_CONFIG", "w") as f:
    yaml.dump(cfg, f)
EOF

echo ""
echo "1) Generating smoke dataset..."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
$CONDA_PYTHON -m src.data.generator --config "$SMOKE_CONFIG" || {
    echo "ERROR: Dataset generation failed"
    exit 1
}

echo ""
echo "2) Preprocessing → splits..."
$CONDA_PYTHON -m src.data.preprocess --raw data/raw --out data/processed --scales configs/scales.yaml || {
    echo "ERROR: Preprocessing failed"
    exit 1
}

echo ""
echo "3) Generating dataset card..."
mkdir -p reports
$CONDA_PYTHON -m src.eval.metrics --processed data/processed --raw data/raw --report reports/DATASET_CARD.json || {
    echo "ERROR: Dataset card generation failed"
    exit 1
}

echo ""
echo "4) Running tests..."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
$CONDA_PYTEST -k "dataset or generator or preprocess or schema or constraints or scaling or split or control or processed or mass" --ignore=tests/test_python.py -v --tb=short || {
    echo "ERROR: Tests failed"
    exit 1
}

echo ""
echo "5) Generating quick check plots..."
mkdir -p docs/figures/wp3_quick_checks
$CONDA_PYTHON scripts/plot_quick_checks.py --raw data/raw --processed data/processed --output docs/figures/wp3_quick_checks --n-cases 3 || {
    echo "WARNING: Plotting failed (non-fatal)"
}

echo ""
echo "=== WP3 Validation Complete ==="
echo "Check reports/DATASET_CARD.json and docs/figures/wp3_quick_checks/"

