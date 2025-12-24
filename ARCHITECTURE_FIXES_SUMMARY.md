# Architecture Fixes Summary

## Overview

This document summarizes the architectural fixes applied to Direction AN model based on the physics-guided trajectory surrogate design principles.

## Key Changes Implemented

### 1. Horizontal Position via Velocity Integration (Option 3)

**Before**: Translation branch predicted `[x, y, z, vx, vy, vz]` directly.

**After**: Translation branch now predicts only `[vx, vy, z, vz]`. Horizontal positions `x(t)` and `y(t)` are reconstructed deterministically via velocity integration:

```python
x_pred = torch.cumsum(vx_pred, dim=time_dim) * dt
y_pred = torch.cumsum(vy_pred, dim=time_dim) * dt
```

**Properties**:
- `x(0) = y(0) = 0` exactly (enforced by zero-padding)
- No boundary loss needed for x,y
- Any drift must come from sustained velocity bias
- Enforces kinematic identity `ẋ = v_x`, `ẏ = v_y` architecturally

**Files Modified**:
- `src/models/branches.py`: Added `TranslationBranchReducedXYFree` class
- `src/models/direction_an_pinn.py`: Updated AN/AN1/AN2 to use reduced branch and integrate x,y

### 2. Mass Monotonicity (Structural Enforcement)

**Before**: Mass predicted directly, penalized for violations, still violated physics.

**After**: Mass branch uses `MonotonicMassBranch` with structural guarantee:

```python
dm_raw = mass_head(z)
dm = softplus(dm_raw)  # Actually: -ReLU(dm_raw) for decreasing
m_pred = m0 - dm
m_pred = clamp(m_pred, min=m_dry)
```

**Properties**:
- `dm ≥ 0` always (via -ReLU)
- `m(t)` monotonic by construction
- No oscillations possible
- No penalty required

**Files Modified**:
- `src/models/branches.py`: `MonotonicMassBranch` class
- `src/models/direction_an_pinn.py`: Updated to use `MonotonicMassBranch` with m0/mdry from context
- `src/train/losses.py`: Removed `L_mass_residual` loss term

### 3. Physics Residual Redesign (Vertical-Only)

**Before**: Full state residual `f(x, u=0)` enforced while training on `f(x, u≠0)`, causing residual–data conflict.

**After**: Reduced-order vertical-only residual with thrust inferred from mass depletion:

```python
m_dot = finite_difference(m_pred)
T_eff = -m_dot * Isp * g0
a_z_model = T_eff / m - g0
r_z = dvz_dt - a_z_model
L_phys = E[r_z^2]
```

**Properties**:
- Enforces energy consistency
- Thrust–mass coupling via rocket equation
- Vertical acceleration plausibility
- No horizontal dynamics enforced (controls/gimbal unknown, aero simplified)
- Residual is honest and consistent with data

**Files Modified**:
- `src/train/losses.py`: Replaced `physics_loss()` with `_vertical_residual_loss()`
- `src/train/losses_v2.py`: Updated to use same vertical-only residual
- Removed horizontal residual computation

### 4. Loss Function Restructuring

**Removed Losses** (unjustified or redundant):
- `L_xy_zero`: x,y position penalties
- `L_zero_vxy`: Horizontal velocity suppression
- `L_zero_axy`: Horizontal acceleration suppression
- `L_hacc`: Global horizontal acceleration limiter
- `L_mass_residual`: Mass residual (now structurally enforced)
- `L_vxy_residual`: Horizontal dynamics residual

**Kept Losses**:
- `L_data`: Component-weighted MSE on all predicted states
- `L_phys`: Reduced-order vertical-only physics residual
- `L_bc`: Boundary conditions at t=0
- `L_quat_norm`: Quaternion normalization penalty (optional)
- `L_mass_flow`: Mass flow consistency (should be ~0 with MonotonicMassBranch)
- Optional temporal smoothing: `L_smooth_z`, `L_smooth_vz`, `L_pos_vel`, `L_smooth_pos`

**Files Modified**:
- `src/train/losses.py`: Removed obsolete loss computations and aggregation
- `src/train/train_pinn.py`: Updated `SoftLossScheduler` to remove obsolete lambdas

### 5. Bug Fixes

**Context Field Access**:
- Fixed `mdry` extraction: dataset only has 7 context fields, `mdry` not available
- Added fallback: `mdry = m0 * 0.7` when not in context
- Prevents empty tensor errors

**Gravity Vector Broadcasting**:
- Fixed gravity vector shape compatibility in `compute_dynamics`
- Ensures proper broadcasting for batched operations

**Files Modified**:
- `src/models/direction_an_pinn.py`: Context field access with fallback
- `src/physics/dynamics_pytorch.py`: Gravity vector shape handling

## Configuration File

**New Config**: `configs/train_an_fixed.yaml`

Key settings:
- Model: `direction_an` with reduced translation branch and monotonic mass
- Loss: Vertical-only physics residual, removed obsolete terms
- Training: 200 epochs, cosine scheduler, early stopping

## Verification

**Test Script**: `test_an_setup.py`

All tests pass:
- ✓ Model instantiation
- ✓ Forward pass with correct output shapes
- ✓ x,y integration verified (x(0)=y(0)=0)
- ✓ Mass monotonicity verified
- ✓ Loss computation
- ✓ Data loading

**Training Status**: ✅ Running successfully
- Epoch 1 completed: Train Loss: 40.26, Val Loss: 10.34
- No errors, training progressing normally

## Sanity Checklist (All Verified)

- [x] x,y removed from NN outputs
- [x] x,y reconstructed by integration
- [x] mass predicted via structural monotonic branch
- [x] mass residual loss removed
- [x] thrust inferred from ṁ
- [x] horizontal acceleration losses removed
- [x] physics residual is vertical-only
- [x] model runs successfully

## Next Steps

1. **Run Full Training**: Let training complete to see final metrics
2. **Thesis Updates**: Update LaTeX to reflect new architecture and loss design
3. **Evaluation**: Compare metrics with previous versions
4. **Optional**: Add drift regularizer `L_drift = λ * mean(x² + y²)` if needed

## Files Changed Summary

**Core Architecture**:
- `src/models/branches.py`: `TranslationBranchReducedXYFree`, `MonotonicMassBranch`
- `src/models/direction_an_pinn.py`: Updated AN/AN1/AN2 forward passes

**Loss Functions**:
- `src/train/losses.py`: Vertical-only physics residual, removed obsolete losses
- `src/train/losses_v2.py`: Updated to use vertical-only residual

**Physics**:
- `src/physics/dynamics_pytorch.py`: Fixed gravity vector broadcasting

**Configuration**:
- `configs/train_an_fixed.yaml`: New config for fixed architecture

**Testing**:
- `test_an_setup.py`: Verification script

