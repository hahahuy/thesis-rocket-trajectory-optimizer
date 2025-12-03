# Phase Schedule Explanation

## Overview

The phase schedule (`phase_schedule` in loss config) implements a **two-phase training strategy** for soft physics and regularization losses. This helps the model learn basic patterns first, then refine with physics constraints.

## How It Works

### Phase 1: Data-Focused Training (Epochs 0 to `phase_start`)

**Duration**: First `phase1_ratio * total_epochs` epochs
- Example: If `phase1_ratio: 0.55` and `epochs: 160`, Phase 1 = epochs 0-88

**What happens**:
- **Soft physics losses are set to ZERO**:
  - `lambda_mass_residual = 0.0`
  - `lambda_vz_residual = 0.0`
  - `lambda_vxy_residual = 0.0`
  - `lambda_smooth_z = 0.0`
  - `lambda_smooth_vz = 0.0`
  - `lambda_pos_vel = 0.0`
  - `lambda_smooth_pos = 0.0`
  - `lambda_zero_vxy = 0.0`
  - `lambda_zero_axy = 0.0`
  - `lambda_hacc = 0.0`
  - `lambda_xy_zero = 0.0`

- **Only core losses are active**:
  - `lambda_data` (data fitting)
  - `lambda_phys` (physics ODE residual)
  - `lambda_bc` (boundary conditions)

**Purpose**: Let the model learn the basic trajectory patterns from data without being constrained by soft physics penalties.

### Phase 2: Physics Refinement (Epochs `phase_start` to `total_epochs`)

**Duration**: Remaining epochs after `phase_start`
- Example: Epochs 89-160 (if `phase1_ratio: 0.55`)

**What happens**:
- Soft physics losses **ramp up** from 0.0 to their target values
- Ramp type: `cosine` (smooth) or `linear` (linear)

**Ramp Formula** (cosine):
```python
progress = (epoch - phase_start) / (total_epochs - phase_start)
scale = 0.5 * (1.0 - cos(π * progress))
lambda_actual = lambda_target * scale
```

**Example** (cosine ramp):
- Epoch 89 (start of Phase 2): `scale ≈ 0.0` → losses still near zero
- Epoch 124 (middle): `scale ≈ 0.5` → losses at 50% of target
- Epoch 160 (end): `scale = 1.0` → losses at full target values

**Purpose**: Gradually introduce physics constraints to refine the model without disrupting learned patterns.

## Implementation Details

### Code Location
- **Scheduler**: `SoftLossScheduler` class in `src/train/train_pinn.py` (lines 140-205)
- **Update**: Called in training loop: `soft_loss_scheduler.update(epoch)` (line 819)

### Loss Keys Affected
The scheduler modifies these loss function attributes:
```python
self.keys = [
    "lambda_mass_residual",
    "lambda_vz_residual",
    "lambda_vxy_residual",
    "lambda_smooth_z",
    "lambda_smooth_vz",
    "lambda_pos_vel",
    "lambda_smooth_pos",
    "lambda_zero_vxy",
    "lambda_zero_axy",
    "lambda_hacc",
    "lambda_xy_zero",
]
```

### Early Stopping
- **Phase 1**: Uses `early_stopping_patience` (default: 25)
- **Phase 2**: Uses `early_stopping_patience_phase2` (default: 40, if > phase1)
- This gives Phase 2 more patience since it's refining, not learning from scratch

## Configuration Example

```yaml
loss:
  # Target values (what losses will reach by end of Phase 2)
  lambda_mass_residual: 0.025
  lambda_vz_residual: 0.025
  lambda_vxy_residual: 0.005
  lambda_pos_vel: 0.5
  
  phase_schedule:
    enabled: true           # Enable two-phase training
    phase1_ratio: 0.55      # Phase 1 = first 55% of epochs
    ramp: cosine            # Smooth cosine ramp (or "linear")
```

## Why This Works

1. **Phase 1**: Model learns basic trajectory shapes from data
   - No physics constraints → model can explore freely
   - Focuses on fitting data and satisfying ODE physics

2. **Phase 2**: Model refines with physics-aware constraints
   - Gradual ramp prevents sudden disruption
   - Soft physics losses guide toward more physically consistent solutions
   - Cosine ramp is smoother than linear (better for stability)

## Current Status (v2 Configs)

In the v2 configs (`train_direction_d153_v2.yaml`, `train_an_v2.yaml`):
- **Phase schedule**: Enabled with `phase1_ratio: 0.55`, `ramp: cosine`
- **Target weights**: Reduced compared to v1 (less aggressive regularization)
- **Horizontal suppression**: Disabled (`lambda_zero_vxy = 0.0`, etc.) - v2 features (T_mag, q_dyn) should help with this naturally

## Notes

- The phase schedule **only affects soft physics losses**, not core losses (data, physics ODE, BC)
- If `enabled: false`, all losses use their target values from epoch 0
- The ramp ensures smooth transition - no sudden jumps in loss landscape

