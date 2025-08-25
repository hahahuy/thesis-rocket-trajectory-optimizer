pinn-rocket-trajectory/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── setup.cfg
├── .gitignore
├── Makefile
├── docker/
│ ├── Dockerfile
│ └── docker-compose.yml
├── configs/
│ ├── default.yaml
│ ├── train.yaml
│ └── optimize.yaml
├── data/
│ ├── raw/ # raw solver outputs, keep read-only
│ ├── processed/ # normalized, split datasets (HDF5 / npz)
│ └── README.md # description of dataset format and sources
├── src/
│ ├── __init__.py
│ ├── physics/ # deterministic physics & dynamics
│ │ ├── __init__.py
│ │ ├── dynamics.py # ODE definitions and helpers
│ │ ├── atmosphere.py # density, wind models
│ │ └── constraints.py # dynamic pressure, path constraints
│ ├── solver/ # baseline optimal control code
│ │ ├── __init__.py
│ │ ├── collocation.py # CasADi wrappers, direct collocation setup
│ │ ├── shooting.py # alternative solvers and wrappers
│ │ └── utils.py # solver helpers, initial guess generators
│ ├── data/ # ETL and dataset utilities
│ │ ├── __init__.py
│ │ ├── generator.py # parameter sweep + trajectory generation
│ │ ├── preprocess.py # normalization, splits, storage helpers
│ │ └── storage.py # HDF5/NPZ dataset read/write helpers
│ ├── models/ # PINN and hybrid models
│ │ ├── __init__.py
│ │ ├── pinn.py # PINN model class, loss composition
│ │ ├── residual_net.py # hybrid residual model
│ │ └── architectures.py# MLP blocks, Fourier features, utilities
│ ├── train/ # training script modules
│ │ ├── __init__.py
│ │ ├── train_pinn.py # training loop for PINN, logging hooks
│ │ ├── train_residual.py
│ │ └── callbacks.py # LR schedulers, early stopping, L-BFGS wrapper
│ ├── optim/ # optimization pipeline using surrogate
│ │ ├── __init__.py
│ │ ├── parameterize.py # control knot parameterization utilities
│ │ ├── optimize_with_surrogate.py # differentiable loop using autodiff
│ │ └── cma_es_wrapper.py # fallback optimizer wrapper
│ ├── experiments/ # experiment orchestration helpers
│ │ ├── run_experiment.py
│ │ └── reproduce_figure.py
│ ├── eval/ # evaluation and metrics
│ │ ├── __init__.py
│ │ ├── metrics.py # RMSE, terminal cost diff, constraint violation
│ │ └── uq.py # ensemble and MC dropout helpers
│ └── utils/ # general utilities
│ ├── __init__.py
│ ├── io.py # config loader, checkpoint IO
│ ├── logging.py # logging and experiment id generator
│ └── tests_utils.py # small helpers for unit tests
├── notebooks/
│ ├── 00-overview.ipynb # quickstart, minimal example reproducing fig 1
│ ├── 01-data-generation.ipynb
│ ├── 02-train-pinn.ipynb
│ └── 03-optimize-using-surrogate.ipynb
├── experiments/
│ ├── exp_2025-09-01_baseline/ # named experiment directories
│ │ ├── config.yaml
│ │ ├── checkpoints/
│ │ └── logs/
│ └── exp_.../
├── scripts/
│ ├── gen_data.sh
│ ├── train_pinn.sh
│ ├── optimize.sh
│ └── evaluate.sh
├── tests/
│ ├── test_dynamics.py
│ ├── test_pinn_loss.py
│ └── test_optimizer_gradients.py
└── docs/
├── design.md
├── figures/
└── thesis_notes.md