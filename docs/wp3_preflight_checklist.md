# WP3 → WP4 Pre-Flight Checklist

This document describes the automated validation and visual checks for WP3 before proceeding to WP4 (PINN training).

## Quick Start

Run the full pre-flight validation:

```bash
make -f Makefile.wp3 validate_wp3
```

This will:
1. Generate a small smoke dataset (12 cases)
2. Preprocess and create splits
3. Generate dataset card
4. Run all tests
5. Generate quick check plots

## Automated Gates

### 1. Contracts Locked
- ✅ `solve_ocp` stub in `src/solver/collocation.py` with `SolveResult` contract
- ✅ `integrate_truth` stub in `src/physics/dynamics.py` with `IntegrateResult` contract
- ✅ State order `[x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, wx,wy,wz, m]` validated

**Tests:** `test_schema_consistency.py`, `test_control_unit_vector.py`

### 2. Dataset Health & Feasibility
- ✅ Success rate ≥ 90-95% (after retries)
- ✅ 0 hard violations of `q ≤ qmax` and `n ≤ nmax`
- ✅ Quaternion unit error < 1e-6 (auto-renormalized)
- ✅ No NaNs/Infs
- ✅ Mass monotonic when thrust > 0
- ✅ Uniform time grid (identical N per case)

**Tests:** `test_constraints_clean.py`, `test_mass_monotonic.py`, `test_processed_shapes.py`

### 3. Scaling Sanity
- ✅ `scales.yaml` includes `L, V, T, M, F, W (=1/T)`
- ✅ Round-trip error < 1e-9
- ✅ Angular rates scaled by `W`
- ✅ Quaternions unscaled
- ✅ Context vector normalized (physics-aware)

**Tests:** `test_scaling_roundtrip.py`

### 4. Schema Consistency
- ✅ HDF5 keys with correct shapes/dtypes
- ✅ Checksums computed and stored

**Tests:** `test_schema_consistency.py`

### 5. Reproducibility Metadata
- ✅ `git_hash`, `config_hash`, `seed`, `created_utc` in all files
- ✅ `DATASET_CARD.json` with stats, ranges, fail rate

**Output:** `reports/DATASET_CARD.json`

### 6. CI/Tests
All tests pass:
- `test_generator_success_rate.py`
- `test_constraints_clean.py`
- `test_scaling_roundtrip.py`
- `test_schema_consistency.py`
- `test_split_stratification.py`
- `test_control_unit_vector.py`
- `test_processed_shapes.py`
- `test_mass_monotonic.py`

## Visual Checks

Plots are generated in `docs/figures/wp3_quick_checks/`:

1. **Trajectory slices** (`trajectory_slices.png`): z(t), v_y(t), mass(t), q_dyn(t), n_load(t)
2. **Parameter coverage** (`parameter_histograms.png`, `parameter_scatter.png`): histograms and 2D scatter
3. **Quaternion sanity** (`quaternion_sanity.png`): norm and components
4. **Knots vs truth** (`knots_vs_truth_*.png`): OCP knots overlaid on integrated trajectory

## Commands

```bash
# Full validation
make -f Makefile.wp3 validate_wp3

# Just tests
make -f Makefile.wp3 test

# Just plots
make -f Makefile.wp3 plot

# Manual steps
python -m src.data.generator --config configs/dataset.yaml
python -m src.data.preprocess --raw data/raw --out data/processed --scales configs/scales.yaml
python -m src.eval.metrics --processed data/processed --raw data/raw --report reports/DATASET_CARD.json
python scripts/plot_quick_checks.py --raw data/raw --processed data/processed --output docs/figures/wp3_quick_checks
pytest -k "dataset or generator or preprocess" -v
```

## Go/No-Go Criteria

| Gate | Go if... | No-Go if... |
|------|----------|-------------|
| Feasibility | 0 violations; success ≥90% | Violations or success <90% |
| Scaling | Round-trip <1e-9; ang-rate uses W | Quat scaled; wrong scaling |
| Schema | All keys & shapes OK | Missing/mismatched |
| Reproducibility | git/config/seed logged | Missing/inconsistent |
| Splits | Stratified & balanced | Skewed distributions |

## Known Traps

- **Double scaling**: Keep SI end-to-end; scale only in `preprocess`
- **Control not unit-norm**: Assert `||uT||=1` in integrator
- **Inconsistent state order**: Test equivalence
- **Context vector drift**: Freeze `CONTEXT_FIELDS` order
- **Angle wrapping**: Consider `(sin, cos)` pairs if discontinuities appear

## Exit Stamp

After validation, you should have:
- ✅ `data/processed/{train,val,test}.h5`
- ✅ `data/processed/DATASET_CARD.json`
- ✅ `reports/DATASET_CARD.json`
- ✅ `docs/figures/wp3_quick_checks/*.png`
- ✅ All tests green

