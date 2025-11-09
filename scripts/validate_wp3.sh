#!/usr/bin/env bash
# WP3 pre-flight validation script
set -euo pipefail

echo "=== WP3 Pre-Flight Validation ==="

# Configuration
CONFIG="${1:-configs/dataset.yaml}"
SMOKE_N="${2:-12}"  # Small smoke test size

echo "Using config: $CONFIG"
echo "Smoke test size: $SMOKE_N"

# Create smoke config
SMOKE_CONFIG="configs/dataset_smoke.yaml"
python3 <<EOF
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
python -m src.data.generator --config "$SMOKE_CONFIG" || {
    echo "ERROR: Dataset generation failed"
    exit 1
}

echo ""
echo "2) Preprocessing → splits..."
python -m src.data.preprocess --raw data/raw --out data/processed --scales configs/scales.yaml || {
    echo "ERROR: Preprocessing failed"
    exit 1
}

echo ""
echo "3) Generating dataset card..."
mkdir -p reports
python -m src.eval.metrics --processed data/processed --raw data/raw --report reports/DATASET_CARD.json || {
    echo "ERROR: Dataset card generation failed"
    exit 1
}

echo ""
echo "4) Running tests..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest -k "dataset or generator or preprocess or schema or constraints or scaling or split or control or processed or mass" --ignore=tests/test_python.py -v --tb=short || {
    echo "ERROR: Tests failed"
    exit 1
}

echo ""
echo "5) Generating quick check plots..."
mkdir -p docs/figures/wp3_quick_checks
python scripts/plot_quick_checks.py --raw data/raw --processed data/processed --output docs/figures/wp3_quick_checks --n-cases 3 || {
    echo "WARNING: Plotting failed (non-fatal)"
}

echo ""
echo "=== WP3 Validation Complete ==="
echo "Check reports/DATASET_CARD.json and docs/figures/wp3_quick_checks/"

