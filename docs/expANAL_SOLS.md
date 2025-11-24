# C3 Architecture Implementation Guide
## Enhanced Hybrid PINN with RMSE Reduction Solutions

**Date:** 2025-XX-XX  
**Base Architecture:** C2 (Shared Stem + Dedicated Branches)  
**Purpose:** Address RMSE root causes through architectural improvements

---

## Table of Contents

1. [Overview](#overview)
   - [Motivation](#motivation)
   - [C3 Architecture Philosophy](#c3-architecture-philosophy)
2. [Problem Analysis](#problem-analysis)
   - [Experiment Results Summary](#experiment-results-summary)
   - [Root Cause Analysis](#root-cause-analysis)
     - [1. Vertical Dynamics (z, vz) - Highest Errors](#1-vertical-dynamics-z-vz---highest-errors)
     - [2. Quaternion Errors (Rotation RMSE: 0.38)](#2-quaternion-errors-rotation-rmse-038)
     - [3. Mass Flow Violations (4.2% increases!)](#3-mass-flow-violations-42-increases)
3. [C2 vs C3 Architecture Comparison](#c2-vs-c3-architecture-comparison)
   - [High-Level Comparison](#high-level-comparison)
   - [Complete Architecture Flow Comparison](#complete-architecture-flow-comparison)
     - [C2 Architecture Flow](#c2-architecture-flow)
     - [C3 Architecture Flow (Complete System)](#c3-architecture-flow-complete-system)
4. [Complete C3 System Structure](#complete-c3-system-structure)
   - [Module Hierarchy](#module-hierarchy)
   - [Data Flow Dimensions](#data-flow-dimensions)
5. [Solution Implementation Details](#solution-implementation-details)
   - [Solution 1: Physics-Informed Vertical Dynamics Branch](#solution-1-physics-informed-vertical-dynamics-branch)
   - [Solution 2: Quaternion Minimal Representation](#solution-2-quaternion-minimal-representation)
   - [Solution 3: Structural Mass Monotonicity](#solution-3-structural-mass-monotonicity)
   - [Solution 4: Higher-Order ODE Integration (RK4)](#solution-4-higher-order-ode-integration-rk4)
   - [Solution 5: Cross-Branch Coordination](#solution-5-cross-branch-coordination)
   - [Solution 6: Enhanced z0 Initialization](#solution-6-enhanced-z0-initialization)
6. [Integration Guide](#integration-guide)
   - [Step-by-Step Implementation](#step-by-step-implementation)
     - [Phase 1: High Priority Solutions (Part 1)](#phase-1-high-priority-solutions-part-1)
     - [Phase 2: Medium Priority Solutions (Part 2)](#phase-2-medium-priority-solutions-part-2)
     - [Phase 3: Low Priority Solution](#phase-3-low-priority-solution)
   - [Complete C3 Class Structure](#complete-c3-class-structure)
   - [Configuration File](#configuration-file)
   - [Training Integration](#training-integration)
7. [Expected Performance](#expected-performance)
   - [RMSE Improvements by Component](#rmse-improvements-by-component)

---

## Overview

### Motivation

**C2 Baseline Performance (exp3)**:
- Total RMSE: **0.96**
- Translation RMSE: 1.41
- Rotation RMSE: **0.38** (3.5x worse than exp2)
- Mass RMSE: 0.19
- **Key Issues**: Quaternion norm=1.08, high vertical dynamics errors (z: 0.91-1.10, vz: 2.98-3.45)

**Failed Approach (exp4 - Loss Weighting)**:
- Total RMSE: **1.005** (worse than C2!)
- Mass violations: **4.2%** (physically impossible)
- **Conclusion**: Loss weighting doesn't fix root causes, only penalizes errors

**C3 Solution**: Architectural improvements that address root causes structurally, not through penalties.

### C3 Architecture Philosophy

C3 = C2 + 6 Architectural Solutions that:
1. **Enforce physics constraints structurally** (mass monotonicity, quaternion unit norm)
2. **Reduce integration errors** (RK4 vs Euler)
3. **Explicit physics computation** (altitude→density→drag)
4. **Cross-branch coordination** (aerodynamic coupling)
5. **Better initialization** (hybrid physics+data z0)

**Key Principle**: Fix root causes in architecture, not through loss penalties.

---

## Problem Analysis

### Experiment Results Summary

| Metric | exp1 (PINN) | exp2 (Sequence) | exp3 (C2) | exp4 (Weighted) | **C3 Target** |
|--------|-------------|-----------------|-----------|-----------------|---------------|
| **Total RMSE** | 0.84 ✅ | 0.86 | 0.96 | **1.005** ❌ | **0.60-0.75** |
| **Translation RMSE** | 1.27 | 1.30 | 1.41 | 1.48 | **0.90-1.20** |
| **Rotation RMSE** | 0.11 ✅ | **0.047** ✅✅ | 0.38 ❌ | 0.38 ❌ | **0.15-0.25** |
| **Mass RMSE** | 0.14 | 0.16 | 0.19 | 0.14 | **0.10-0.12** |
| **Mass Violations** | - | - | - | **4.2%** ❌ | **0%** ✅ |
| **Quaternion Norm** | 0.86 | - | 1.08 | 1.01 | **1.0** ✅ |

### Root Cause Analysis

#### 1. Vertical Dynamics (z, vz) - Highest Errors

**Physics Chain**:
```
vz_dot = (T/m) - g - drag_z/m
```
Where:
- `T` (thrust) ∝ `m` (mass decreases)
- `drag_z` ∝ `rho(z)` × `|v|²` × `Cd`
- `rho(z)` = `rho0 * exp(-z/H)` (exponential decay)
- `g` constant, competes with time-varying forces

**C2 Problems**:
1. **Information bottleneck**: Shared Stem compresses 1501×49D → 128D, loses altitude-dependent physics
2. **Integration errors**: Euler method O(dt) accumulates over 1501 steps
3. **No explicit physics**: Model must learn `rho(z)` from data
4. **Complex coupling**: Altitude→density→drag→acceleration chain not explicit

**RMSE Impact**: z: 0.91-1.10, vz: 2.98-3.45 (highest single component error)

#### 2. Quaternion Errors (Rotation RMSE: 0.38)

**C2 Problems**:
1. **Normalization gradient issues**: `q_norm = q / ||q||` is non-differentiable
2. **Masking effect**: Model learns wrong quaternions, relies on normalization to fix
3. **Coupling with translation**: Aerodynamics depend on orientation, but branches independent
4. **Limited z0 window**: Only 10 steps used, may miss initial quaternion state

**RMSE Impact**: Rotation RMSE 0.38 (3.5x worse than exp2's 0.047), quaternion norm 0.86-1.08

#### 3. Mass Flow Violations (4.2% increases!)

**C2 Problems**:
1. **No architectural constraint**: Mass branch predicts independently, no `m(t+1) <= m(t)` guarantee
2. **Integration error**: Euler can produce positive `m_dot` → mass increases
3. **Circular dependency**: Mass → thrust → trajectory → mass (model doesn't know thrust)

**RMSE Impact**: 4.2% of time steps have physically impossible mass increases

---

## C2 vs C3 Architecture Comparison

### High-Level Comparison

| Component | C2 (Current) | C3 (Enhanced) | Solution |
|-----------|--------------|---------------|----------|
| **Mass Branch** | Predicts Δm (can be positive) | `MonotonicMassBranch`: Always ≤ 0 | Solution 3 |
| **Rotation Branch** | Predicts quaternion, normalizes | `RotationBranchMinimal`: Rotation vector → quaternion | Solution 2 |
| **ODE Integration** | Euler (1st-order, O(dt)) | RK4 (4th-order, O(dt⁴)) | Solution 4 |
| **Translation Branch** | Standard MLP | `PhysicsAwareTranslationBranch`: Explicit density/drag | Solution 1 |
| **Branch Coordination** | Independent | `CoordinatedBranches`: Aerodynamic coupling | Solution 5 |
| **z0 Initialization** | Window (10 steps) | `EnhancedZ0Derivation`: Full sequence + physics | Solution 6 |
| **Quaternion** | Normalized externally | Always unit norm (from minimal rep) | Solution 2 |
| **Mass** | Can increase (4.2% violations) | Structurally guaranteed decreasing | Solution 3 |

### Complete Architecture Flow Comparison

#### C2 Architecture Flow

```
Inputs: (t [batch,N,1], context [batch,7], initial_state [batch,14])
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ SHARED STEM                                                  │
│                                                              │
│ TimeEmbedding (Fourier, 8 freq) → t_emb [batch,N,17]      │
│ DeepContextEncoder → ctx_emb [batch,N,32]                  │
│ Concatenate → [batch,N,49]                                  │
│ InputProj(49→128) → [batch,N,128]                          │
│ TransformerEncoder(4 layers) → shared_emb [batch,N,128]   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ z0 DERIVATION (Limited Window)                              │
│                                                              │
│ Extract window: shared_emb[:, :10, :] → [batch,10,128]    │
│ SequenceEncoder → [batch,10,128]                           │
│ Mean pooling → [batch,128]                                 │
│ z0_proj(128→64) → z0 [batch,64]                            │
│                                                              │
│ Problem: Only 6.7% of trajectory used                      │
└──────────────┬──────────────────────────────────────────────┘
               │ z0
               ▼
┌─────────────────────────────────────────────────────────────┐
│ LATENT ODE (Euler Integration)                              │
│                                                              │
│ For i = 0 to N-1:                                          │
│   dz_dt = dynamics_net(z_i, cond_i)                        │
│   z_{i+1} = z_i + dt * dz_dt  (Euler step)                 │
│                                                              │
│ Problem: O(dt) error accumulates over 1501 steps           │
└──────────────┬──────────────────────────────────────────────┘
               │ z_traj [batch,N,64]
               ▼
┌──────────┬──────────┬──────────┐
│ Trans    │ Rotation │ Mass     │
│ Branch   │ Branch   │ Branch   │
│ [64→128→ │ [64→256→ │ [64→64→  │
│ 128→6]   │ 256→7]   │ 1]      │
│          │          │          │
│ [x,y,z,  │ [q0,q1,  │ [Δm]     │
│ vx,vy,vz]│ q2,q3,   │          │
│          │ wx,wy,wz]│          │
└────┬─────┴────┬─────┴────┬─────┘
     │          │          │
     │          ▼          │
     │    normalize_quat() │
     │    (non-diff!)      │
     │          │          │
     └──────────┴──────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ Δ-STATE RECONSTRUCTION                                      │
│                                                              │
│ state_delta = [translation || rotation || mass_delta]       │
│ state = initial_state + state_delta                        │
│                                                              │
│ Problems:                                                   │
│ - Mass can increase (4.2% violations)                      │
│ - Quaternion norm errors (0.86-1.08)                       │
│ - No physics-aware corrections                             │
│ - No branch coordination                                    │
└──────────────┬──────────────────────────────────────────────┘
               │ state [batch,N,14]
               ▼
          Loss Function
```

#### C3 Architecture Flow (Complete System)

```
Inputs: (t [batch,N,1], context [batch,7], initial_state [batch,14])
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ SHARED STEM (Unchanged from C2)                            │
│                                                              │
│ TimeEmbedding → t_emb [batch,N,17]                         │
│ DeepContextEncoder → ctx_emb [batch,N,32]                 │
│ TransformerEncoder → shared_emb [batch,N,128]            │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ ENHANCED z0 DERIVATION (Solution 6)                         │
│                                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐         │
│ │ Data-Driven z0       │  │ Physics-Informed z0  │         │
│ │                      │  │                      │         │
│ │ Full sequence mean:  │  │ PhysicsEncoder:      │         │
│ │ z0_data_full =       │  │ z0_physics =         │         │
│ │   shared_emb.mean() │  │   encode(context,    │         │
│ │                      │  │            s0)       │         │
│ │ Window + Transformer:│ │                      │         │
│ │ z0_data_window =    │  │                      │         │
│ │   Transformer(      │  │                      │         │
│ │     shared_emb[:10])│  │                      │         │
│ │                      │  │                      │         │
│ │ z0_data = 0.7*full + │  │                      │         │
│ │          0.3*window │  │                      │         │
│ └──────┬───────────────┘  └──────┬───────────────┘         │
│        │                         │                         │
│        └──────────┬───────────────┘                         │
│                   │                                         │
│                   ▼                                         │
│        z0 = 0.3*z0_physics + 0.7*z0_data                    │
│                   │                                         │
│        Better initialization → Less error propagation      │
└──────────────┬──────────────────────────────────────────────┘
               │ z0 [batch,64]
               ▼
┌─────────────────────────────────────────────────────────────┐
│ RK4 LATENT ODE (Solution 4)                                 │
│                                                              │
│ For i = 0 to N-1:                                          │
│   k1 = f(z_i, cond_i)                                       │
│   k2 = f(z_i + dt/2*k1, cond_mid)                          │
│   k3 = f(z_i + dt/2*k2, cond_mid)                          │
│   k4 = f(z_i + dt*k3, cond_end)                            │
│   z_{i+1} = z_i + dt/6*(k1+2*k2+2*k3+k4)                   │
│                                                              │
│ Error: O(dt⁴) vs Euler's O(dt) - ~1000x smaller!          │
└──────────────┬──────────────────────────────────────────────┘
               │ z_traj [batch,N,64] (more accurate)
               ▼
┌─────────────────────────────────────────────────────────────┐
│ COORDINATED BRANCHES (Solutions 1,2,3,5)                    │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ INITIAL PREDICTIONS                                     ││
│ │                                                         ││
│ │ ┌──────────┬──────────┬──────────┐                    ││
│ │ │ Physics- │ Rotation │ Monotonic│                    ││
│ │ │ Aware    │ Minimal  │ Mass     │                    ││
│ │ │ Trans    │ (Sol 2)  │ (Sol 3)  │                    ││
│ │ │ (Sol 1)  │          │          │                    ││
│ │ └────┬─────┴────┬─────┴────┬─────┘                    ││
│ │      │          │          │                           ││
│ │      │          │          │                           ││
│ │      ▼          ▼          ▼                           ││
│ │ translation  rotation   mass                            ││
│ │ [x,y,z,      [q,w]      [m]                             ││
│ │  vx,vy,vz]   (unit q)   (decreasing)                    ││
│ └──────────────┬─────────────────────────────────────────┘│
│                │                                           │
│                ▼                                           │
│ ┌───────────────────────────────────────────────────────┐│
│ │ AERODYNAMIC COUPLING MODULE (Solution 5)              ││
│ │                                                         ││
│ │ Extract: |v|, q, z (altitude)                         ││
│ │ Compute: rho(z) = rho0 * exp(-z/H)                    ││
│ │ Compute: drag_correction = f(|v|, q, rho, context)    ││
│ │ Apply: translation += drag_correction                  ││
│ └──────────────┬──────────────────────────────────────────┘│
│                │                                           │
│     translation_corrected                                  │
│     rotation (unit quaternion from minimal rep)           │
│     mass (monotonically decreasing)                       │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHYSICS-AWARE CORRECTION (Solution 1 - Applied in Branch)  │
│                                                              │
│ Extract: z (altitude), |v| (velocity magnitude)            │
│ Compute: rho(z) = rho0 * exp(-z/H)  (explicit physics)    │
│ Compute: drag = 0.5 * rho * |v|² * Cd * S                 │
│ Correct: vz_corrected = vz - drag_z / m                    │
│                                                              │
│ Result: Explicit altitude→density→drag chain                │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ Δ-STATE RECONSTRUCTION                                      │
│                                                              │
│ state_delta = [translation_corrected || rotation || mass]  │
│ state = initial_state + state_delta                        │
│                                                              │
│ Guarantees:                                                 │
│ - Mass always decreases (structural)                        │
│ - Quaternion always unit norm (from minimal rep)            │
│ - Physics-aware corrections applied                         │
│ - Branches coordinated through aerodynamics                 │
└──────────────┬──────────────────────────────────────────────┘
               │ state [batch,N,14]
               ▼
          Loss Function
```

---

## Complete C3 System Structure

### Module Hierarchy

```
RocketHybridPINNC3
│
├── SharedStem (unchanged from C2)
│   ├── TimeEmbedding (FourierFeatures)
│   ├── DeepContextEncoder
│   └── TransformerEncoder
│
├── EnhancedZ0Derivation (Solution 6)
│   ├── PhysicsInformedZ0Encoder
│   ├── DataDrivenZ0Encoder (Transformer-based)
│   └── Blending: z0 = α*z0_physics + (1-α)*z0_data
│
├── LatentDynamicsNet (unchanged)
│
├── LatentODEBlockRK4 (Solution 4)
│   └── RK4 integration (k1, k2, k3, k4 stages)
│
└── CoordinatedBranches (Solutions 1,2,3,5)
    ├── PhysicsAwareTranslationBranch (Solution 1)
    │   ├── Standard MLP
    │   └── PhysicsComputationLayer
    │       ├── compute_density(z)
    │       └── compute_drag_force(rho, |v|, context)
    │
    ├── RotationBranchMinimal (Solution 2)
    │   ├── RotationVectorMLP → [rx, ry, rz]
    │   ├── rotation_vector_to_quaternion() → [q0,q1,q2,q3]
    │   └── AngularVelocityMLP → [wx, wy, wz]
    │
    ├── MonotonicMassBranch (Solution 3)
    │   ├── MassDeltaMLP → mass_delta_raw
    │   ├── -ReLU(mass_delta_raw) → mass_delta (always ≤ 0)
    │   └── cumsum(mass_delta) + m0 → mass (monotonically decreasing)
    │
    └── AerodynamicCouplingModule (Solution 5)
        ├── Extract: |v|, q, rho
        ├── Compute: drag_correction = f(|v|, q, rho, context)
        └── Apply: translation += drag_correction
```

### Data Flow Dimensions

```
Input Dimensions:
  t: [batch, N, 1]           (N=1501 typically)
  context: [batch, 7]        (m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag)
  initial_state: [batch, 14]  (x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz,m)

Shared Stem:
  t_emb: [batch, N, 17]       (1 + 2×8 Fourier features)
  ctx_emb: [batch, N, 32]    (DeepContextEncoder output)
  shared_emb: [batch, N, 128] (Shared Stem output)

z0 Derivation:
  z0_data_full: [batch, 64]   (Full sequence mean)
  z0_data_window: [batch, 64] (Window + Transformer)
  z0_physics: [batch, 64]     (Physics-informed)
  z0: [batch, 64]             (Blended)

Latent ODE:
  z_traj: [batch, N, 64]      (RK4 integrated)

Branches:
  translation: [batch, N, 6]  (x,y,z,vx,vy,vz)
  rotation: [batch, N, 7]      (q0,q1,q2,q3,wx,wy,wz)
  mass: [batch, N, 1]          (m)

Output:
  state: [batch, N, 14]       (Final state trajectory)
```

---

## Solution Implementation Details

### Solution 3: Structural Mass Monotonicity

**RMSE Root Cause Addressed**: Mass violations (4.2% → 0%)

**Architecture Change**:
```python
# C2: MassBranch
mass_delta = self.mlp(z_latent)  # Can be positive!
mass = initial_mass + cumsum(mass_delta)  # Can increase ❌

# C3: MonotonicMassBranch
mass_delta_raw = self.mlp(z_latent)  # [batch, N, 1]
mass_delta = -torch.relu(mass_delta_raw)  # Always ≤ 0 ✅
mass = torch.cumsum(mass_delta, dim=1) + m0  # Always decreasing ✅
```

**Direct RMSE Impact**:
- **Mass violations**: 4.2% → 0% (100% elimination)
- **Mass RMSE**: 0.14-0.19 → 0.10-0.12 (20-30% reduction)
- **Physics consistency**: Structural guarantee, no penalty needed

**File**: `src/models/branches.py` - Add `MonotonicMassBranch` class

---

### Solution 2: Quaternion Minimal Representation

**RMSE Root Cause Addressed**: Quaternion normalization errors (norm 0.86-1.08 → always 1.0)

**Architecture Change**:
```python
# C2: RotationBranch
quat_raw = self.mlp(z_latent)[..., :4]  # [batch, N, 4]
quat_norm = normalize_quaternion(quat_raw)  # q / ||q|| (non-diff!) ❌

# C3: RotationBranchMinimal
rotation_vec = self.rotation_vec_mlp(z_latent)  # [batch, N, 3]
quaternion = rotation_vector_to_quaternion(rotation_vec)  # Always ||q||=1 ✅
```

**Rotation Vector to Quaternion Conversion**:
```python
def rotation_vector_to_quaternion(rotation_vec):
    """
    Convert rotation vector (axis-angle) to quaternion.
    Always produces unit quaternion by construction.
    
    angle = ||rotation_vec||
    axis = rotation_vec / angle
    q0 = cos(angle/2)
    q1,q2,q3 = sin(angle/2) * axis
    """
    angle = torch.linalg.norm(rotation_vec, dim=-1, keepdim=True)
    axis = rotation_vec / (angle + 1e-8)
    half_angle = angle / 2.0
    q0 = torch.cos(half_angle)
    q_vec = torch.sin(half_angle) * axis
    quaternion = torch.cat([q0, q_vec], dim=-1)
    # Normalize for numerical stability
    quat_norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True) + 1e-8
    return quaternion / quat_norm
```

**Direct RMSE Impact**:
- **Quaternion norm**: 0.86-1.08 → Always 1.0 (100% fix)
- **Rotation RMSE**: 0.38 → 0.15-0.20 (40-60% reduction)
- **Gradient stability**: No non-differentiable normalization
- **No masking**: Model learns correct quaternions directly

**File**: `src/models/branches.py` - Add `RotationBranchMinimal` class and `rotation_vector_to_quaternion()` function

---

### Solution 4: Higher-Order ODE Integration (RK4)

**RMSE Root Cause Addressed**: Integration error accumulation (O(dt) → O(dt⁴))

**Architecture Change**:
```python
# C2: LatentODEBlock (Euler)
for i in range(N-1):
    dz_dt = dynamics_net(z_i, cond_i)
    z_{i+1} = z_i + dt * dz_dt  # O(dt) error ❌

# C3: LatentODEBlockRK4
for i in range(N-1):
    k1 = dynamics_net(z_i, cond_i)
    k2 = dynamics_net(z_i + dt/2*k1, cond_mid)
    k3 = dynamics_net(z_i + dt/2*k2, cond_mid)
    k4 = dynamics_net(z_i + dt*k3, cond_end)
    z_{i+1} = z_i + dt/6*(k1+2*k2+2*k3+k4)  # O(dt⁴) error ✅
```

**Direct RMSE Impact**:
- **Integration error**: ~1000x smaller per step (O(dt⁴) vs O(dt))
- **Vertical RMSE**: z: 0.91-1.10 → 0.60-0.80, vz: 2.98-3.45 → 1.80-2.40 (30-50% reduction)
- **Long-term stability**: Errors don't accumulate as fast
- **All components benefit**: Better z0 evolution → better all predictions

**File**: `src/models/latent_ode.py` - Add `LatentODEBlockRK4` class

**Computational Cost**: 4x function evaluations per step (vs Euler's 1), but much more accurate

---

### Solution 1: Physics-Informed Vertical Dynamics Branch

**RMSE Root Cause Addressed**: Vertical dynamics errors (z: 0.91-1.10, vz: 2.98-3.45)

**Architecture Change**:
```python
# C2: TranslationBranch
translation = self.mlp(z_latent)  # [x,y,z,vx,vy,vz]
# Must learn rho(z) from data ❌

# C3: PhysicsAwareTranslationBranch
translation_raw = self.mlp(z_latent)  # Initial prediction
z = translation_raw[..., 2:3]  # Extract altitude
rho = rho0 * exp(-z/H)  # Explicit physics computation ✅
v_mag = ||translation_raw[..., 3:6]||
drag = 0.5 * rho * v_mag² * Cd * S
vz_corrected = vz - drag / m  # Physics-aware correction ✅
```

**Physics Computation Layer**:
```python
class PhysicsComputationLayer(nn.Module):
    def compute_density(self, altitude, context):
        """rho(z) = rho0 * exp(-z/H)"""
        H = context.get('H', 8400.0)
        rho0 = context.get('rho0', 1.225)
        return rho0 * torch.exp(-torch.clamp(altitude, min=0.0) / H)
    
    def compute_drag_force(self, rho, v_mag, context):
        """F_drag = 0.5 * rho * |v|² * Cd * S"""
        Cd = context.get('Cd', 0.3)
        S_ref = context.get('S_ref', 1.0)
        q = 0.5 * rho * v_mag ** 2
        return q * Cd * S_ref
```

**Direct RMSE Impact**:
- **Vertical RMSE**: z: 0.91-1.10 → 0.60-0.80 (20-30% reduction), vz: 2.98-3.45 → 1.80-2.40 (30-40% reduction)
- **Explicit physics**: No need to learn `rho(z)` from data
- **Reduces bottleneck**: Density computed directly, not compressed
- **Better consistency**: Drag explicitly computed from altitude and velocity

**File**: `src/models/branches.py` - Add `PhysicsAwareTranslationBranch` class  
**File**: `src/models/physics_layers.py` - New file with `PhysicsComputationLayer` class

**Note**: Creates dependency on mass. Options:
- Use mass from `MonotonicMassBranch` (predict mass first)
- Use initial mass estimate
- Iterative refinement

---

### Solution 5: Cross-Branch Coordination

**RMSE Root Cause Addressed**: Rotation-translation coupling errors (rotation RMSE 0.38)

**Architecture Change**:
```python
# C2: Independent Branches
translation = translation_branch(z_traj)  # Independent
rotation = rotation_branch(z_traj)        # Independent
mass = mass_branch(z_traj)                # Independent
# No coordination ❌

# C3: CoordinatedBranches
translation_init = translation_branch(z_traj)
rotation_init = rotation_branch(z_traj)
mass = mass_branch(z_traj)

# Extract coupling variables
v_mag = ||translation_init[..., 3:6]||
q = rotation_init[..., :4]
rho = compute_density(z_altitude, context)

# Compute aerodynamic coupling
drag_correction = coupling_module(|v|, q, rho, context)
translation_corrected = translation_init + drag_correction  # Coordinated ✅
```

**Aerodynamic Coupling Module**:
```python
class AerodynamicCouplingModule(nn.Module):
    def compute_drag_coupling(self, v_mag, quaternion, rho, context):
        """
        Input: [|v|, q0, q1, q2, q3, rho] → [batch, N, 6]
        Output: drag correction to translation [dx, dy, dz, dvx, dvy, dvz]
        """
        coupling_input = torch.cat([v_mag, quaternion, rho], dim=-1)
        drag_correction = self.coupling_net(coupling_input)  # [batch, N, 6]
        return drag_correction
```

**Direct RMSE Impact**:
- **Rotation RMSE**: 0.38 → 0.25-0.30 (20-35% reduction)
- **Translation RMSE**: 1.27-1.48 → 1.10-1.30 (10-15% reduction)
- **Explicit coupling**: Drag computed from both velocity and orientation
- **Addresses root cause**: Translation errors no longer cascade to rotation

**File**: `src/models/coordination.py` - New file with `CoordinatedBranches` and `AerodynamicCouplingModule` classes

---

### Solution 6: Enhanced z0 Initialization

**RMSE Root Cause Addressed**: Limited encoder window (10 steps → full sequence + physics)

**Architecture Change**:
```python
# C2: Limited Window
window = shared_emb[:, :10, :]  # Only 6.7% of trajectory
z0_tokens = TransformerEncoder(window).mean(dim=1)
z0 = z0_proj(z0_tokens)  # [batch, 64]
# Information bottleneck ❌

# C3: EnhancedZ0Derivation
# Option A: Full sequence mean
z0_data_full = shared_emb.mean(dim=1)  # Use all N steps ✅

# Option B: Window + Transformer (better encoding)
z0_data_window = TransformerEncoder(shared_emb[:, :10, :]).mean(dim=1)

# Option C: Physics-informed
z0_physics = PhysicsEncoder(context, initial_state)  # From known physics ✅

# Blend
z0_data = 0.7*z0_data_full + 0.3*z0_data_window
z0 = 0.3*z0_physics + 0.7*z0_data  # Hybrid approach ✅
```

**Direct RMSE Impact**:
- **Better initialization**: Reduces error propagation through ODE
- **Improves all components**: Better z0 → better entire trajectory
- **Marginal but consistent**: 5-10% reduction across all RMSE components
- **Training stability**: Faster convergence

**File**: `src/models/z0_encoder.py` - New file with `EnhancedZ0Derivation` and `PhysicsInformedZ0Encoder` classes

---

## Integration Guide

### Step-by-Step Implementation

#### Phase 1: High Priority Solutions (Part 1)

**Step 1.1: Implement MonotonicMassBranch**
```python
# File: src/models/branches.py
# Add after existing MassBranch class

class MonotonicMassBranch(nn.Module):
    """Structural mass monotonicity - Solution 3"""
    def __init__(self, hidden_dim: int = 64, branch_dims: list = [64]):
        # ... (see Part 1 details)
    
    def forward(self, z_latent, m0):
        mass_delta_raw = self.mlp(z_latent)
        mass_delta = -torch.relu(mass_delta_raw)  # Always ≤ 0
        mass = torch.cumsum(mass_delta, dim=1) + m0.unsqueeze(1)
        return mass
```

**Step 1.2: Implement RotationBranchMinimal**
```python
# File: src/models/branches.py
# Add rotation_vector_to_quaternion() function
# Add RotationBranchMinimal class

def rotation_vector_to_quaternion(rotation_vec):
    # ... (see Part 1 details)

class RotationBranchMinimal(nn.Module):
    """Quaternion minimal representation - Solution 2"""
    # ... (see Part 1 details)
```

**Step 1.3: Implement LatentODEBlockRK4**
```python
# File: src/models/latent_ode.py
# Add after existing LatentODEBlock class

class LatentODEBlockRK4(nn.Module):
    """RK4 integration - Solution 4"""
    # ... (see Part 1 details)
```

**Step 1.4: Update RocketHybridPINNC2 → Create RocketHybridPINNC3**
```python
# File: src/models/hybrid_pinn.py
# Copy RocketHybridPINNC2, rename to RocketHybridPINNC3
# Replace components:

class RocketHybridPINNC3(nn.Module):
    def __init__(self, ...):
        # ... (same as C2)
        
        # Replace branches
        self.mass_branch = MonotonicMassBranch(...)  # Solution 3
        self.rotation_branch = RotationBranchMinimal(...)  # Solution 2
        
        # Replace ODE
        self.ode_block = LatentODEBlockRK4(self.dynamics_net)  # Solution 4
        
        # Remove quaternion normalization (not needed with minimal rep)
```

**Test Phase 1**: Should see:
- Mass violations: 4.2% → 0%
- Quaternion norm: 1.08 → 1.0
- Better vertical RMSE (from RK4)

#### Phase 2: Medium Priority Solutions (Part 2)

**Step 2.1: Create Physics Layers**
```python
# File: src/models/physics_layers.py (new file)
class PhysicsComputationLayer(nn.Module):
    # ... (see Part 2 details)
```

**Step 2.2: Implement PhysicsAwareTranslationBranch**
```python
# File: src/models/branches.py
class PhysicsAwareTranslationBranch(nn.Module):
    """Physics-aware vertical dynamics - Solution 1"""
    # ... (see Part 2 details)
```

**Step 2.3: Create Coordination Module**
```python
# File: src/models/coordination.py (new file)
class AerodynamicCouplingModule(nn.Module):
    # ... (see Part 2 details)

class CoordinatedBranches(nn.Module):
    # ... (see Part 2 details)
```

**Step 2.4: Update C3 to Use Coordinated Branches**
```python
# In RocketHybridPINNC3.__init__()
self.branches = CoordinatedBranches(
    translation_branch=PhysicsAwareTranslationBranch(...),
    rotation_branch=RotationBranchMinimal(...),
    mass_branch=MonotonicMassBranch(...),
    coupling_module=AerodynamicCouplingModule(...),
    physics_layer=PhysicsComputationLayer()
)
```

**Test Phase 2**: Should see:
- Better translation-rotation coupling
- Improved vertical dynamics RMSE

#### Phase 3: Low Priority Solution

**Step 3.1: Create z0 Encoder**
```python
# File: src/models/z0_encoder.py (new file)
class PhysicsInformedZ0Encoder(nn.Module):
    # ... (see Part 2 details)

class EnhancedZ0Derivation(nn.Module):
    # ... (see Part 2 details)
```

**Step 3.2: Update C3 z0 Derivation**
```python
# In RocketHybridPINNC3.__init__()
self.z0_derivation = EnhancedZ0Derivation(...)

# In forward() method
z0 = self.z0_derivation(shared_emb, context, initial_state, t_emb, ctx_emb)
```

**Test Phase 3**: Should see marginal but consistent improvement across all components

### Complete C3 Class Structure

```python
class RocketHybridPINNC3(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture]
    Enhanced C2 with all RMSE reduction solutions.
    
    Solutions Integrated:
    - Solution 3: MonotonicMassBranch (structural mass constraint)
    - Solution 2: RotationBranchMinimal (quaternion minimal representation)
    - Solution 4: LatentODEBlockRK4 (higher-order integration)
    - Solution 1: PhysicsAwareTranslationBranch (explicit physics)
    - Solution 5: CoordinatedBranches (aerodynamic coupling)
    - Solution 6: EnhancedZ0Derivation (hybrid initialization)
    """
    
    requires_initial_state = True
    
    def __init__(self, ...):
        # Shared Stem (unchanged from C2)
        self.shared_stem = SharedStem(...)
        
        # Enhanced z0 derivation (Solution 6)
        self.z0_derivation = EnhancedZ0Derivation(...)
        
        # Latent dynamics (unchanged)
        self.dynamics_net = LatentDynamicsNet(...)
        
        # RK4 ODE (Solution 4)
        self.ode_block = LatentODEBlockRK4(self.dynamics_net)
        
        # Coordinated branches (Solutions 1, 2, 3, 5)
        self.branches = CoordinatedBranches(
            translation_branch=PhysicsAwareTranslationBranch(...),  # Sol 1
            rotation_branch=RotationBranchMinimal(...),              # Sol 2
            mass_branch=MonotonicMassBranch(...),                    # Sol 3
            coupling_module=AerodynamicCouplingModule(...),         # Sol 5
            physics_layer=PhysicsComputationLayer()
        )
    
    def forward(self, t, context, initial_state):
        # 1. Shared Stem
        shared_emb = self.shared_stem(t, context)
        
        # 2. Enhanced z0 (Solution 6)
        t_emb = self.shared_stem.time_embedding(t)
        ctx_emb = self.shared_stem.context_encoder(context)
        z0 = self.z0_derivation(shared_emb, context, initial_state, t_emb, ctx_emb)
        
        # 3. RK4 ODE (Solution 4)
        condition = torch.cat([t_emb, ctx_emb], dim=-1)
        z_traj = self.ode_block(z0, t, condition)
        
        # 4. Coordinated branches (Solutions 1, 2, 3, 5)
        translation, rotation, mass = self.branches(z_traj, context, initial_state)
        
        # 5. Δ-state reconstruction
        state_delta = torch.cat([translation, rotation, mass], dim=-1)
        state = initial_state.unsqueeze(1) + state_delta
        
        return state
```

### Configuration File

**File**: `configs/model_C3.yaml`
```yaml
model:
  type: hybrid_c3
  
  # Same as C2
  latent_dim: 64
  fourier_features: 8
  shared_stem_hidden_dim: 128
  temporal_type: transformer
  temporal_n_layers: 4
  temporal_n_heads: 4
  temporal_dim_feedforward: 512
  encoder_window: 10
  translation_branch_dims: [128, 128]
  rotation_branch_dims: [256, 256]
  mass_branch_dims: [64]
  dynamics_n_hidden: 3
  dynamics_n_neurons: 128
  activation: tanh
  transformer_activation: gelu
  layer_norm: true
  dropout: 0.05
  debug_stats: true
  
  # C3-specific parameters
  z0_blend_alpha: 0.3  # Weight for physics-informed z0 (0.3 = 30% physics, 70% data)
  use_rk4: true        # Use RK4 instead of Euler
  use_physics_aware_translation: true
  use_coordinated_branches: true
```

### Training Integration

**File**: `src/train/train_pinn.py`
```python
# Add model_type: hybrid_c3 branch
elif model_type == "hybrid_c3":
    from src.models.hybrid_pinn import RocketHybridPINNC3
    
    model = RocketHybridPINNC3(
        context_dim=context_dim,
        # ... (same parameters as C2)
        z0_blend_alpha=float(model_cfg.get("z0_blend_alpha", 0.3)),
        use_rk4=bool(model_cfg.get("use_rk4", True)),
        # ...
    ).to(device)
```

---

## Expected Performance

### RMSE Improvements by Component

| Component | C2 (exp3) | C3 (Expected) | Improvement | Solution(s) |
|-----------|-----------|---------------|-------------|-------------|
| **Total RMSE** | 0.96 | **0.60-0.75** | 25-40% | All solutions |
| **Translation RMSE** | 1.41 | **0.90-1.20** | 15-30% | Solutions 1, 4, 5 |
| **Rotation RMSE** | 0.38 | **0.15-0.25** | 35-60% | Solutions 2, 5 |
| **Mass RMSE** | 0.19 | **0.10-0.12** | 20-30% | Solution 3 |
| **Vertical (z)** | 0.91-1.10 | **0.60-0.80** | 20-30% | Solutions 1, 4 |
| **Vertical (vz)** | 2.98-3.45 | **1.80-2.40** | 30-40% | Solutions 1, 4 |
| **Mass Violations** | - | **0%** | 100% fix | Solution 3 |
| **Quaternion Norm** | 1.08 | **1.0** | 100% fix | Solution 2 |
