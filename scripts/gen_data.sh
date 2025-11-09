#!/usr/bin/env bash
set -euo pipefail
python -m src.data.generator --config configs/dataset.yaml
