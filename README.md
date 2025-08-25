# Rocket Trajectory Optimization with Physics-Informed Neural Networks

This repository contains the implementation of physics-informed neural networks (PINNs) for rocket trajectory optimization, combining traditional optimal control methods with machine learning approaches.

## Project Structure

- `src/physics/` - Deterministic physics & dynamics models
- `src/solver/` - Baseline optimal control solvers
- `src/models/` - PINN and hybrid neural network models
- `src/train/` - Training scripts and utilities
- `src/optim/` - Optimization pipeline using surrogate models
- `src/data/` - Data generation and preprocessing utilities
- `notebooks/` - Jupyter notebooks for exploration and examples
- `experiments/` - Experiment configurations and results

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Generate training data: `python scripts/gen_data.sh`
3. Train PINN model: `python scripts/train_pinn.sh`
4. Run optimization: `python scripts/optimize.sh`

## Documentation

See `docs/` directory for detailed documentation and design notes.
