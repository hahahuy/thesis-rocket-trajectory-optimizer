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

## Dataset

**Dataset version: wp3_final** (tag: `wp3_final`)

The processed dataset is available in `data/processed/`:
- `train.h5` (120 samples)
- `val.h5` (20 samples)  
- `test.h5` (20 samples)

Dataset card: `reports/DATASET_CARD.json`

## Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/hahahuy/thesis-rocket-trajectory-optimizer.git
cd thesis-rocket-trajectory-optimizer
```

2. **Install environment**

See **[docs/SETUP.md](docs/SETUP.md)** for complete setup instructions.

Quick start:
```bash
# Python deps (editable install)
pip install -e .

# Core packages if needed
pip install casadi numpy scipy matplotlib h5py pyyaml pytest pytest-cov

# Optional: test which linear solvers are available for IPOPT
python3 scripts/test_linear_solvers.py
```

**Tip**: Prefer MUMPS by default; HSL/MA97 gives best performance if installed. See **[docs/SETUP_HSL_MUMPS.md](docs/SETUP_HSL_MUMPS.md)** for details.

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

### Quick Start
* **[docs/SETUP.md](docs/SETUP.md)**: Main setup guide with quick start
* **[docs/SETUP_ENVIRONMENT.md](docs/SETUP_ENVIRONMENT.md)**: Complete environment setup (Python, C++, Windows, CI/CD)
* **[docs/SETUP_HSL_MUMPS.md](docs/SETUP_HSL_MUMPS.md)**: HSL/MUMPS linear solver setup

### Architecture & Design
* **[docs/DESIGN.md](docs/DESIGN.md)**: High-level system architecture and design overview
* **[docs/architecture_diagram.md](docs/architecture_diagram.md)**: Mermaid diagrams for all PINN architectures
* **[docs/ARCHITECTURE_CHANGELOG.md](docs/ARCHITECTURE_CHANGELOG.md)**: PINN architecture evolution history

### Work Package Documentation
* **[docs/wp1_comprehensive_description.md](docs/wp1_comprehensive_description.md)**: Physics core (6-DOF dynamics library)
* **[docs/wp2_comprehensive_description.md](docs/wp2_comprehensive_description.md)**: Optimal control baseline (CasADi/IPOPT solver)
* **[docs/wp3_comprehensive_description.md](docs/wp3_comprehensive_description.md)**: Dataset generation and preprocessing
* **[docs/wp4_comprehensive_description.md](docs/wp4_comprehensive_description.md)**: PINN training and models

### Results & Analysis
* **[docs/RESULTS_AND_VALIDATION.md](docs/RESULTS_AND_VALIDATION.md)**: Validation results (WP1-4) and experiment summaries (exp1-5)
* **[docs/expANAL_SOLS.md](docs/expANAL_SOLS.md)**: C3 architecture implementation guide

### Documentation Guide
* **[docs/README.md](docs/README.md)**: Complete documentation structure and navigation guide

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
