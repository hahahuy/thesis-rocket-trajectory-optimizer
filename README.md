# Rocket Trajectory Optimizer

This repository implements a 6-DOF rocket physics core (WP1) and an optimal control baseline (WP2) using direct collocation with CasADi/IPOPT. It also includes scripts for validation, benchmarking, robustness, and documentation for future PINN-based work.

---

## Overview

* **Dynamics**: 6-DOF rocket model with translational and rotational states, thrust vectoring, aerodynamic forces/moments, wind, and dynamic pressure (q) limits.
* **Baseline**: Direct collocation optimal control (CasADi/IPOPT) provides reference solutions.
* **Surrogate**: PINN or hybrid residual model trained with both solver data and physics residuals.
* **Optimizer**: Surrogate embedded in a differentiable optimization loop; compared against collocation baseline.
* **Evaluation**: Accuracy, speedup, robustness to uncertainties, uncertainty quantification, and ablation studies.

---

## Project Structure (high level)

* `src/physics/` — C++ 6-DOF physics library (dynamics, constraints, smooth funcs)
* `src/solver/` — Python WP2 solver (CasADi collocation, constraints, utils)
* `src/utils/` — C++ utils (scaling, reproducibility) and Python helpers
* `scripts/` — Validation, benchmarking, robustness, plotting
* `configs/` — YAML configs: `phys.yaml`, `limits.yaml`, `scales.yaml`, `ocp.yaml`
* `tests/` — C++ physics tests and Python WP2 tests
* `build/` — CMake build outputs (ignored)
* `experiments/` — Outputs from validation/benchmarks/robustness
* `docs/` — Design + comprehensive WP1/WP2 documentation

---

## Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/hahahuy/thesis-rocket-trajectory-optimizer.git
cd thesis-rocket-trajectory-optimizer
```

2. **Install environment (Python + IPOPT deps)**

```bash
# Python deps (editable install)
pip install -e .

# Core packages if needed
pip install casadi numpy scipy matplotlib h5py pyyaml pytest pytest-cov

# Optional: test which linear solvers are available for IPOPT
python3 scripts/test_linear_solvers.py
```

Tip: Prefer MUMPS by default; HSL/MA97 gives best performance if installed. See the HSL/MUMPS section in `docs/wp2_comprehensive_description.md`.

3. **Build WP1 (C++ physics demos/validation)**

```bash
mkdir -p build; cd build
cmake ..
make -j$(nproc) validate_dynamics
```

4. **Run validation and generate reference CSVs**

```bash
./build/validate_dynamics --all
```

5. **WP2 (OCP baseline) — run with Make**

```bash
make -f Makefile.wp2 test; make -f Makefile.wp2 validate; make -f Makefile.wp2 benchmark-quick; make -f Makefile.wp2 robustness
```

6. **Optional quick scripts (if present)**

```bash
# Example scripts
bash scripts/benchmark_solver.py --quick
python3 scripts/plot_trajectory.py --input experiments/wp2_validation.json || true
```

---

## Documentation

* WP1 (physics core): `docs/wp1_comprehensive_description.md`
* WP2 (OCP baseline): `docs/wp2_comprehensive_description.md`
  - Includes: operations guide (how to run), and HSL/MUMPS installation notes
* Design overview: `docs/design.md`
* Setup: `docs/setup_guide.md`, `docs/QUICK_START_HSL.md`
* CI/setup: `docs/ci_setup.md`

---

## Key Commands

```bash
# WP1 build + validation
cd build; cmake ..; make -j$(nproc) validate_dynamics; cd -; ./build/validate_dynamics --all

# WP2 tasks
make -f Makefile.wp2 test; make -f Makefile.wp2 validate; make -f Makefile.wp2 benchmark-quick; make -f Makefile.wp2 robustness

# Coverage (Python WP2)
make -f Makefile.wp2 coverage; xdg-open htmlcov/index.html || true
```

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.

---

## Citation

If you use this repository, please cite the thesis and this repo (BibTeX to be provided after defense).
