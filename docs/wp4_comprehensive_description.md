# WP4 — Physics-Informed Neural Network (PINN) and Hybrid Surrogate Modeling

## 🎯 Objective

Train a **physics-informed neural network (PINN)** (and optionally, a hybrid residual variant) that learns to reproduce 6-DOF rocket ascent trajectories under varying parameters. The model must:

* Honor physical laws (via residual loss terms).
* Generalize across sampled parameter space (`m0, Isp, Cd, Tmax, wind…`).
* Serve later (in WP5) as a differentiable **surrogate model** inside an optimization loop.

---

## 🧩 Structure Overview

```
src/
  models/
    ├── pinn.py                # pure physics-informed model
    ├── residual_net.py        # hybrid residual model (Δ-state correction)
    ├── architectures.py       # MLP blocks, Fourier/time embeddings, layer norms
  train/
    ├── train_pinn.py          # main training loop
    ├── train_residual.py
    ├── losses.py              # physics/data/boundary loss composition
    ├── callbacks.py           # LR schedulers, early stopping, checkpoints
  eval/
    ├── metrics.py             # RMSE, MAE, residual diagnostics
    ├── visualize_pinn.py      # trajectory comparison plots
  utils/
    ├── loaders.py             # dataset + DataLoader builders
    ├── scaling.py             # to_nd/from_nd helpers (already done)
    └── seed.py                # seed control for reproducibility
  physics/
    └── dynamics_pytorch.py    # PyTorch dynamics for autograd
configs/
  ├── train.yaml
  └── model.yaml
scripts/
  ├── train_pinn.sh
  ├── train_residual.sh
  └── evaluate_pinn.sh
notebooks/
  └── 04-pinn-training.ipynb   # demo for presentation/thesis figures
```

---

## ⚙️ Implementation Details

### **1️⃣ Data Loading and Preprocessing**

**Implementation:** `src/utils/loaders.py`

The `RocketDataset` class loads processed HDF5 files with structure:
- `inputs/t`: [n_cases, N] time grid (nondimensional)
- `inputs/context`: [n_cases, context_dim] context parameters (normalized)
- `targets/state`: [n_cases, N, 14] states (nondimensional)

**Features:**
- Supports time subsampling for faster training
- Case-based sampling (randomize cases, keep time order within case)
- Automatic scaling from metadata

**Usage:**
```python
from src.utils.loaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data/processed",
    batch_size=8,
    time_subsample=None  # or e.g., 5 for every 5th point
)
```

---

### **2️⃣ Model Architecture**

#### 2.1 Base PINN Network

**Implementation:** `src/models/pinn.py`

Architecture:
- **Input:** `(t, context)` where:
  - `t`: Time [..., 1] (nondimensional)
  - `context`: Context vector [..., context_dim] (normalized)
- **Time Embedding:** Fourier features `[t, sin(2πk t), cos(2πk t)]` for k=1..K
- **Context Encoder:** MLP to embed context parameters
- **MLP Network:** 6 hidden layers × 128 neurons, tanh activation
- **Output:** State [..., 14] (nondimensional)

**Configuration:** `configs/model.yaml`
```yaml
model:
  type: pinn
  n_hidden: 6
  n_neurons: 128
  activation: tanh
  fourier_features: 8
  layer_norm: true
  dropout: 0.05
```

#### 2.2 Residual Hybrid Option

**Implementation:** `src/models/residual_net.py`

Predicts residual corrections to a baseline trajectory:
```
ŝ(t) = s̃(t) + Δs(t)
```

Where `s̃(t)` is from a classical integrator (baseline) and `Δs(t)` is the network prediction.

**Benefits:**
- Faster convergence
- Better stability
- Can leverage existing integrator code

---

### **3️⃣ Loss Function Design**

**Implementation:** `src/train/losses.py`

Total loss:
```
L = λ_data * L_data + λ_phys * L_phys + λ_bc * L_bc
```

#### 3.1 Data Loss
Mean-squared error between predicted and ground-truth normalized states:
```python
L_data = mse(pred_state, true_state)
```

#### 3.2 Physics Loss
Residuals of the governing ODEs, using autograd:
```python
r = s_dot_numerical - f(s_pred, u, p)
L_phys = mean_square(r)
```

Where `f` is the PyTorch dynamics function (`src/physics/dynamics_pytorch.py`).

#### 3.3 Boundary Loss
Penalize mismatch at t=0:
```python
L_bc = ||ŝ(0) - s(0)||²
```

#### 3.4 Weight Scheduling

**Implementation:** `src/train/callbacks.py` → `LossWeightScheduler`

Supports:
- **Fixed:** Constant weights
- **Linear:** Linear interpolation from init to final
- **Exponential:** Exponential growth/decay

**Configuration:** `configs/train.yaml`
```yaml
loss:
  lambda_data: 1.0
  lambda_phys: 0.1
  lambda_bc: 1.0
  lambda_phys_final: 1.0
  homotopy_schedule: linear  # linear | exponential | fixed
```

---

### **4️⃣ Training Loop**

**Implementation:** `src/train/train_pinn.py`

**Core Structure:**
```python
for epoch in range(epochs):
    # Train
    for batch in train_loader:
        pred = model(batch["t"], batch["context"])
        loss, loss_dict = loss_fn(pred, batch["state"], batch["t"])
        loss.backward()
        optimizer.step()
    
    # Validate
    val_loss = validate(model, val_loader, loss_fn)
    
    # Update scheduler, checkpoints, early stopping
    scheduler.step()
    checkpoint_callback.save(...)
    if early_stopping(val_loss):
        break
```

**Features:**
- Gradient clipping (max_norm=1.0)
- Learning rate scheduling (cosine, plateau, step, exponential)
- Early stopping based on validation loss
- Automatic checkpointing (best, last, periodic)
- Loss weight scheduling (homotopy)

**Optimizer:**
- Adam (LR = 1e-3), `weight_decay=1e-5`
- Scheduler: cosine decay (default)

---

### **5️⃣ Validation and Diagnostics**

**Implementation:** `src/eval/visualize_pinn.py`

#### Quantitative Metrics

- **RMSE(state):** Per-component and total
- **Physics residual norm:** Mean and max
- **Terminal state error:** Altitude, velocity, mass

#### Visual Diagnostics

- **Trajectory overlays:** Predicted vs true (altitude, velocity, mass, q_dyn)
- **Residual histograms:** Distribution of physics residuals
- **Loss curves:** Training and validation loss over epochs

**Usage:**
```python
from src.eval.visualize_pinn import evaluate_model, plot_trajectory_comparison

metrics = evaluate_model(model, test_loader, device, scales)
plot_trajectory_comparison(t, pred, true, scales, save_path="fig.png")
```

---

### **6️⃣ PyTorch Dynamics Module**

**Implementation:** `src/physics/dynamics_pytorch.py`

Provides differentiable dynamics computation for physics loss:
- Mirrors CasADi implementation (`src/solver/dynamics_casadi.py`)
- Uses PyTorch tensors for autograd
- Handles batched and unbatched inputs
- Supports nondimensional states

**Key Functions:**
- `compute_dynamics(x, u, params, scales)`: Compute state derivative
- `DynamicsModule`: PyTorch module wrapper

---

## 🚀 Usage

### Training

**Basic training:**
```bash
./scripts/train_pinn.sh --config configs/train.yaml
```

**With custom config:**
```bash
python -m src.train.train_pinn \
    --config configs/train.yaml \
    --data_dir data/processed \
    --experiment_dir experiments \
    --seed 42
```

**Resume from checkpoint:**
```bash
./scripts/train_pinn.sh --resume experiments/pinn_baseline/checkpoints/best.pt
```

### Evaluation

```bash
./scripts/evaluate_pinn.sh \
    --checkpoint experiments/pinn_baseline/checkpoints/best.pt \
    --output_dir evaluation_results
```

### Configuration

Create a merged config file:
```yaml
# configs/pinn_config.yaml
model:
  type: pinn
  n_hidden: 6
  n_neurons: 128
  # ... (see configs/model.yaml)

train:
  experiment_name: pinn_baseline
  batch_size: 8
  epochs: 100
  # ... (see configs/train.yaml)

loss:
  lambda_data: 1.0
  lambda_phys: 0.1
  lambda_bc: 1.0
  homotopy_schedule: linear

physics_config: configs/phys.yaml
scales_config: configs/scales.yaml
```

---

## ✅ Definition of Done (Exit Criteria)

| Category               | Metric / Artifact                        | Target |
| ---------------------- | ---------------------------------------- | ------ |
| **Training quality**   | RMSE(state) ≤ 2 % (normalized), residual norm ≤ 1e-3  | ✓      |
| **Stability**          | No NaNs, gradient norm < 1000            | ✓      |
| **Convergence**        | Val loss plateau < 1 % fluctuation       | ✓      |
| **Logs**               | Training log JSON with all metrics       | ✓      |
| **Model checkpoints**  | best.pt, last.pt saved                   | ✓      |
| **Evaluation figures** | Trajectory overlays, residual histograms | ✓      |
| **Documentation**      | This document + code comments             | ✓      |
| **Repo tag**           | `wp4_final` created                      | ✓      |

---

## 🧩 Deliverables for WP4

1. **Trained models:**
   - `experiments/exp_*/checkpoints/best.pt`
   - `experiments/exp_*/checkpoints/residual_best.pt` (optional)

2. **Configs:**
   - `configs/model.yaml`
   - `configs/train.yaml`

3. **Reports & Figures:**
   - `experiments/exp_*/train_log.json`
   - `experiments/exp_*/figures/*.png`

4. **Documentation:**
   - `docs/wp4_comprehensive_description.md` (this file)
   - `notebooks/04-pinn-training.ipynb`

---

## 🔬 Ablations & Sensitivity

Recommended ablations to run:

1. **Loss weight sensitivity:**
   - `λ_phys = {0, 0.1, 1, 10}`
   - Record validation RMSE and residual norms

2. **Network depth:**
   - `n_hidden = {4, 6, 8}`
   - Compare training time and final RMSE

3. **Data subset:**
   - `{25%, 50%, 100%}` of training data
   - Assess generalization vs data efficiency

4. **Fourier features:**
   - `n_frequencies = {4, 8, 16}`
   - Impact on high-frequency dynamics

Results should be recorded in `experiments/exp_*/metrics.json` for WP6 analysis.

---

## 📝 Notes

- **Nondimensionalization:** All states and time are nondimensionalized using scales from `configs/scales.yaml`. The model operates entirely in normalized space.

- **Control Input:** For physics loss, control is assumed zero if not provided. In WP5, control will be an input to the model.

- **Baseline for Residual:** The residual network requires a baseline trajectory. Currently uses zero baseline; should be replaced with integrator output in production.

- **Gradient Stability:** Physics loss can produce large gradients. Gradient clipping (max_norm=1.0) is applied to stabilize training.

---

## 🔗 Integration with Other Work Packages

- **WP3:** Uses processed datasets and scaling utilities
- **WP5:** Trained model will serve as surrogate in optimization loop
- **WP6:** Ablation results feed into analysis

---

## 📚 References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

- Karniadakis, G. E., et al. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.

