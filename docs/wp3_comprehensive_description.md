## WP3 — Dataset Generation & Preprocessing (Comprehensive Description)

### Objective
Produce clean, scalable datasets of reference trajectories by sweeping parameter space, solving the WP2 collocation OCP, integrating to a standard time grid, and packaging inputs/targets/metadata for PINN training with emphasis on feasibility, normalization, and reproducibility.

### Implemented Structure
- `src/data/`:
  - `__init__.py`
  - `sampler.py`: `lhs_sample`, `sobol_sample`, `persist_samples_table`
  - `storage.py`: HDF5/NPZ IO helpers, SHA256 checksum
  - `generator.py`: Core generation loop, retries, metadata, multiprocessing `Pool` orchestration, CLI (`python -m src.data.generator`)
  - `preprocess.py`: Scaling (`W` angular-rate included), split packing, CLI (`python -m src.data.preprocess`)
- `src/eval/metrics.py`: Minimal dataset card generator CLI
- `configs/dataset.yaml`: Dataset configuration template per spec
- `scripts/gen_data.sh`, `scripts/make_splits.sh`: End-to-end entry scripts
- `data/README.md`: Data layout and commands

### WP2/WP1 Entrypoints (used by WP3)
- OCP: `src/solver/collocation.solve_ocp(phys, limits, ocp_cfg, scales) -> SolveResult`
- Integrator: `src/physics/dynamics.integrate_truth(x0, t, control_cb, phys, limits, env, ...) -> IntegrateResult`

### Files Added
- `src/data/__init__.py`
- `src/data/sampler.py`
- `src/data/storage.py`
- `src/data/generator.py`
- `src/data/preprocess.py`
- `src/eval/metrics.py`
- `configs/dataset.yaml`
- `scripts/gen_data.sh`
- `scripts/make_splits.sh`
- `data/README.md`

### Public Functions/Objects
- `sampler.lhs_sample(n, bounds, seed)`
- `sampler.sobol_sample(n, bounds, seed)`
- `sampler.persist_samples_table(path, keys, samples)`
- `storage.write_hdf5_case(path, payload, metadata)`
- `storage.write_npz_case(path, payload, metadata)`
- `data.generator.run_generation(cfg_path)`
- `data.preprocess.process_raw_to_splits(raw_dir, processed_dir, scales_path)`
- `eval.metrics.build_card(dataset_path, report_path)`

### Current Status
- **Entrypoint contracts locked in:**
  - `src/solver/collocation.solve_ocp(phys, limits, ocp_cfg, scales) -> SolveResult` (stub with `NotImplementedError`)
  - `src/physics/dynamics.integrate_truth(x0, t, control_cb, phys, limits, env, ...) -> IntegrateResult` (stub with `NotImplementedError`)
- **Phys/limits/env builders:** Match detailed spec (vehicle/aero/inertia, actuation/constraints, gravity/wind with gust support).
- **Sanity checks:** State order validation, control unit vector checks, quaternion renormalization, feasibility gates.
- **Context vector:** Physics-aware normalization with canonical field order; handles variable-length vectors.
- **Scaling:** `configs/scales.yaml` updated with `W` angular-rate scale; `to_nd`/`from_nd` apply `W` correctly.
- **Multiprocessing:** Enabled by default (`spawn` context) via `parallel_workers`.
- **Placeholder trajectories:** Realistic vertical ascent simulation with proper initialization, dynamics, and monitors.
- **Plotting:** Fixed to show `v_z` (vertical velocity) instead of `v_y`; uses raw SI values for parameter plots.

### Remaining Tasks (Next Steps)
1) **Implement WP2 `solve_ocp`:** Use `DirectCollocation` from `transcription.py` and IPOPT solver; return `SolveResult` with SI values.
2) **Implement WP1 `integrate_truth`:** Use `scipy.integrate.solve_ivp` (rk45/rk4) or wrap C++ integrator; return `IntegrateResult` with SI values.
3) **Extend feasibility checks:** Add NaN/Inf guards, mass monotonicity checks, state magnitude bounds.
4) **Add stratified splits:** Bin by parameter ranges (e.g., `m0`, `Cd`, `Isp`, `wind_mag`) and persist detailed `splits.json`.
5) **Expand dataset card:** Include param ranges, solver stats, success/fail rates, quality checks, checksum.
6) **Complete PyTest suite:** `test_constraints_clean.py`, `test_scaling_roundtrip.py`, `test_schema_consistency.py`, `test_split_stratification.py`.

### Notes & Assumptions
- **SI everywhere:** All values exchanged between WP3 and WP2/WP1 are SI. Scaling is applied only in preprocessing.
- **State order:** `[x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, wx,wy,wz, m]` (14 vars) enforced via sanity checks.
- **Control format:** `[T, uTx, uTy, uTz]` with `||uT||=1` enforced via sanity checks.
- **HDF5 schema:** Follows spec (`/time`, `/state`, `/control`, `/monitors/*`, `/ocp/*`, `/meta/*`).
- **Quaternions:** Not scaled; renormalized if norm error > 1e-6.
- **Context vector:** Only includes fields present in params; missing fields set to 0.0; physics-aware normalization.
- **Parallelism:** Multiprocessing by default; Ray can be introduced later without changing `_generate_case` signature.
