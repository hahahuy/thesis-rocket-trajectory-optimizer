#!/usr/bin/env bash
set -euo pipefail
python -m src.data.preprocess --raw data/raw --out data/processed --scales configs/scales.yaml
