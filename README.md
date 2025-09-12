# Rocket Trajectory Optimization with Physics-Informed Neural Networks

This repository implements **physics-informed neural networks (PINNs)** for **rocket trajectory optimization**, combining traditional **optimal control methods** with modern **machine learning** approaches. The goal is to develop a **computationally efficient surrogate** that accelerates trajectory design and enables onboard guidance while remaining physically consistent.

---

## Overview

* **Dynamics**: 6-DOF rocket model with translational and rotational states, thrust vectoring, aerodynamic forces/moments, wind, and dynamic pressure (q) limits.
* **Baseline**: Direct collocation optimal control (CasADi/IPOPT) provides reference solutions.
* **Surrogate**: PINN or hybrid residual model trained with both solver data and physics residuals.
* **Optimizer**: Surrogate embedded in a differentiable optimization loop; compared against collocation baseline.
* **Evaluation**: Accuracy, speedup, robustness to uncertainties, uncertainty quantification, and ablation studies.

---

## Project Structure

* `src/physics/` — Deterministic 6-DOF physics & dynamics models
* `src/solver/` — Baseline optimal control solvers (collocation, shooting)
* `src/models/` — PINN and hybrid neural network models
* `src/train/` — Training scripts and utilities
* `src/optim/` — Optimization pipeline using surrogate models
* `src/data/` — Data generation and preprocessing utilities
* `src/eval/` — Metrics and uncertainty quantification tools
* `src/utils/` — General utilities (IO, logging, config loaders)
* `notebooks/` — Jupyter notebooks for exploration and examples
* `experiments/` — Experiment configurations, results, logs, checkpoints
* `tests/` — Unit tests (dynamics, PINN loss, optimizer gradients)
* `docs/` — Design notes, figures, thesis documentation

---

## Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/hahahuy/thesis-rocket-trajectory-optimizer.git
cd thesis-rocket-trajectory-optimizer
```

2. **Install environment**

```bash
conda env create -f environment.yml
conda activate rocket-pinn
```

3. **Generate baseline data**

```bash
bash scripts/gen_data.sh
```

4. **Train PINN**

```bash
bash scripts/train_pinn.sh
```

5. **Optimize trajectory with surrogate**

```bash
bash scripts/optimize.sh
```

6. **Evaluate results**

```bash
bash scripts/evaluate.sh
```

---

## Documentation

* See `docs/design.md` for detailed design notes.
* See `docs/thesis_notes.md` for research rationale, experiments, and milestones.
* Figures and diagrams are stored in `docs/figures/`.

---

## Repository Structure (Detailed)

```
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── setup.cfg
├── .gitignore
├── Makefile
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   └── optimize.yaml
├── data/
│   ├── raw/        # raw solver outputs (read-only)
│   ├── processed/  # normalized, split datasets (HDF5/NPZ)
│   └── README.md   # dataset format and sources
├── src/
│   ├── __init__.py
│   ├── physics/            # 6-DOF deterministic physics & dynamics
│   │   ├── dynamics.py      # ODE definitions (forces, moments)
│   │   ├── atmosphere.py    # density, wind models
│   │   └── constraints.py   # q-limit, path constraints
│   ├── solver/             # baseline optimal control
│   │   ├── collocation.py   # CasADi collocation wrapper
│   │   ├── shooting.py      # alternative methods
│   │   └── utils.py         # helpers, initial guess gen
│   ├── data/               # ETL and dataset pipeline
│   │   ├── generator.py     # OCP sweeps, trajectory generation
│   │   ├── preprocess.py    # normalization, splits
│   │   └── storage.py       # HDF5/NPZ IO
│   ├── models/             # PINN + residual networks
│   │   ├── pinn.py          # PINN model class, losses
│   │   ├── residual_net.py  # hybrid residual approach
│   │   └── architectures.py # MLP blocks, Fourier features
│   ├── train/              # training loops
│   │   ├── train_pinn.py
│   │   ├── train_residual.py
│   │   └── callbacks.py     # schedulers, early stop, L-BFGS
│   ├── optim/              # optimization with surrogate
│   │   ├── parameterize.py  # control knot parameterization
│   │   ├── optimize_with_surrogate.py
│   │   └── cma_es_wrapper.py
│   ├── experiments/        # orchestration helpers
│   │   ├── run_experiment.py
│   │   └── reproduce_figure.py
│   ├── eval/
│   │   ├── metrics.py       # RMSE, terminal objective, constraint violations
│   │   └── uq.py            # ensemble & MC dropout
│   └── utils/
│       ├── io.py            # config loader, checkpoint IO
│       ├── logging.py       # structured logging
│       └── tests_utils.py
├── notebooks/
│   ├── 00-overview.ipynb
│   ├── 01-data-generation.ipynb
│   ├── 02-train-pinn.ipynb
│   └── 03-optimize-using-surrogate.ipynb
├── experiments/
│   ├── exp_2025-09-01_baseline/
│   │   ├── config.yaml
│   │   ├── checkpoints/
│   │   └── logs/
│   └── exp_.../
├── scripts/
│   ├── gen_data.sh
│   ├── train_pinn.sh
│   ├── optimize.sh
│   └── evaluate.sh
├── tests/
│   ├── test_dynamics.py
│   ├── test_pinn_loss.py
│   └── test_optimizer_gradients.py
└── docs/
    ├── design.md
    ├── figures/
    └── thesis_notes.md
```

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.

---

## Citation

If you use this repository, please cite the thesis and this repo (BibTeX to be provided after defense).
