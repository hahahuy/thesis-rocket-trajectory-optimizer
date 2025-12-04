# Brief #1: Rocket Trajectory Optimization - Data and Model Overview

## Table of Contents

1. [Data Overview](#1-data-overview)
   - 1.1 [Dataset Structure](#11-dataset-structure)
   - 1.2 [Data Pipeline: Raw → Processed v1 → Processed v2](#12-data-pipeline-raw--processed-v1--processed-v2)
     - 1.2.1 [Original/Raw Data](#121-originalraw-data)
     - 1.2.2 [Processed v1/v2 Data](#122-processed-v1v2-data)
   - 1.3 [Exploratory Data Analysis (EDA)](#13-exploratory-data-analysis-eda)
     - 1.3.1 [State Variable Statistics](#131-state-variable-statistics)
     - 1.3.2 [Trajectory Characteristics](#132-trajectory-characteristics)
     - 1.3.3 [Context Parameter Analysis](#133-context-parameter-analysis)
     - 1.3.4 [Data Quality Observations](#134-data-quality-observations)
   - 1.4 [Model Inputs and Outputs](#14-model-inputs-and-outputs)

2. [Loss Function](#2-loss-function)
   - 2.1 [Total Loss Structure](#21-total-loss-structure)
   - 2.2 [Component Losses](#22-component-losses)
   - 2.3 [Loss Weight Scheduling](#23-loss-weight-scheduling)
   - 2.4 [How Loss Function Works](#24-how-loss-function-works)
   - 2.5 [Training Strategy](#25-training-strategy)

3. [Model Architecture](#3-model-architecture)
   - 3.1 [Model Overview](#31-model-overview)
   - 3.2 [Detailed Structure](#32-detailed-structure)
   - 3.3 [How the Model Works](#33-how-the-model-works)

---

## 1. Data Overview

### 1.1 Dataset Structure

**Data Splits:**
- **Train**: 120 cases × 1501 time steps = 180,120 state samples
- **Validation**: 20 cases × 1501 time steps = 30,020 state samples
- **Test**: 20 cases × 1501 time steps = 30,020 state samples
- **Total**: 240,160 state samples

**Time Resolution:**
- Each case contains **1501 time steps**
- Time span: 30 seconds (sampled at 50 Hz)
- Time grid: `t ∈ [0, 30]` seconds (nondimensionalized)

### 1.2 Data Pipeline: Raw → Processed v1 → Processed v2

From generated data as "Raw" → Processed Data v1 / Processed Data v2 to derive more physical parameters

#### 1.2.1 Original/Raw Data

**Location**: `data/raw/`

**Format**: Individual HDF5 files per case: `case_train_*.h5`, `case_val_*.h5`, `case_test_*.h5`

**Complete Schema**:
```
case_*.h5
├── time: [N]                    # Time grid [s] (dimensional, float64)
│   Shape: (1501,) for 30s at 50Hz
│   Range: [0, 30] seconds
├── state: [N, 14]              # State trajectory (dimensional, float64)
│   Shape: (1501, 14)
│   Format: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
│   Units: m, m/s, unit quaternion, rad/s, kg
├── control: [N, 4]             # Control trajectory (dimensional, float64)
│   Shape: (1501, 4)
│   Format: [T, uTx, uTy, uTz]
│   Units: N (thrust magnitude), unitless (thrust direction components)
├── monitors/                    # Monitoring data (optional, float64)
│   ├── q_dyn: [N]              # Dynamic pressure [Pa]
│   ├── n_load: [N]             # Load factor [g]
│   └── rho: [N]                 # Atmospheric density [kg/m³]
├── ocp/                         # OCP solver data (optional, float64)
│   ├── knots/state: [K, 14]    # State at collocation knots
│   │   Shape: (61, 14) for 61 mesh points
│   └── knots/control: [K, 4]   # Control at collocation knots
│       Shape: (61, 4) for 61 mesh points
└── meta/                        # Metadata
    ├── params_used: JSON string # Physical parameters dictionary
    ├── seed: int                # Random seed for reproducibility
    ├── git_hash: string         # Git commit hash
    ├── created_utc: string      # Creation timestamp (ISO format)
    └── checksum: string         # SHA256 checksum of file
```

**Parameter Bounds for Data Generation**:

The dataset is generated using **Latin Hypercube Sampling (LHS)** or **Sobol sequences** across the following parameter space with exact bounds:

| Parameter | Symbol | Range | Unit | Description |
|-----------|--------|-------|------|-------------|
| Initial Mass | `m0` | [45.0, 65.0] | kg | Initial rocket mass |
| Specific Impulse | `Isp` | [220.0, 280.0] | s | Propellant efficiency |
| Drag Coefficient | `Cd` | [0.25, 0.45] | - | Drag coefficient (dimensionless) |
| Lift Curve Slope | `CL_alpha` | [2.5, 4.5] | 1/rad | Lift coefficient per unit angle of attack |
| Pitch Moment Coefficient | `Cm_alpha` | [-1.2, -0.4] | 1/rad | Pitch moment coefficient per unit angle of attack |
| Maximum Thrust | `Tmax` | [3000.0, 5000.0] | N | Maximum available thrust |
| Wind Magnitude | `wind_mag` | [0.0, 15.0] | m/s | Wind speed magnitude (constant wind proxy) |

**Fixed Constraints**:
- Maximum dynamic pressure: `qmax = 40,000 Pa`
- Maximum load factor: `nmax = 5.0 g`

**Fixed/Default Parameters** (held constant during generation):
- Reference area: `S = 0.05 m²`
- Reference length: `l_ref = 1.2 m`
- Moment of inertia (X): `Ix = 10.0 kg·m²`
- Moment of inertia (Y): `Iy = 10.0 kg·m²`
- Moment of inertia (Z): `Iz = 1.0 kg·m²`
- Sea level density: `rho0 = 1.225 kg/m³`
- Atmospheric scale height: `H = 8500.0 m`
- Dry mass: `mdry = 35.0 kg`
- Maximum gimbal angle: `gimbal_max_rad = 0.1745 rad` (~10°)
- Wind direction: `wind_dir_rad = 0.0 rad` (from North)

**Generation Settings**:
- **Sampler**: LHS (Latin Hypercube Sampling) or Sobol sequences
- **Random Seed**: 2025
- **Time Horizon**: 30.0 seconds
- **Grid Frequency**: 50 Hz (uniform grid)
- **Total Cases**: 160 cases (120 train, 20 val, 20 test)
- **OCP Solver**: 61 mesh points, KKT tolerance 1e-6, max iterations 3000

**Characteristics**:
- **Data Format**: Dimensional units (meters, seconds, kilograms, etc.)
- **Case Structure**: Each case file contains a complete trajectory solution from an optimal control problem (OCP) solver
- **State Variables**: 14-dimensional state vector representing 6-DOF rocket dynamics
- **Control Variables**: 4-dimensional control vector (thrust magnitude + 3D thrust direction)

**Data Variation**:
Each case represents a unique combination of the 7 varied physical parameters, resulting in different trajectory characteristics:
- Different altitude profiles (varying maximum altitudes from ~30 to ~42 km)
- Different velocity profiles (varying ascent/descent rates)
- Different mass consumption patterns (fuel burn rates based on Isp and Tmax)
- Different aerodynamic effects (based on Cd, CL_alpha, Cm_alpha)
- Different wind effects (based on wind_mag)

#### 1.2.2 Processed v1/v2 Data

**Locations**: 
- v1: `data/processed/`
- v2: `data/processed_v2/`

**Format (common)**: Consolidated HDF5 files per split: `train.h5`, `val.h5`, `test.h5`

**v1/v2 Base Schema**:
```
train.h5 (or val.h5, test.h5)
├── inputs/
│   ├── t: [n_cases, N]         # Time grid (nondimensional, float64)
│   └── context: [n_cases, context_dim]  # Context parameters (normalized, float64)
├── targets/
│   └── state: [n_cases, N, 14]  # State trajectories (nondimensional, float64)
└── meta/
    ├── scales: JSON string      # Reference scales dictionary
    └── context_fields: JSON array  # Context field names
```

**Concrete Shapes (Direction D1.5.4 dataset)**:
- `t`: 
  - Train: `[120, 1501]`
  - Val/Test: `[20, 1501]`
  - Normalization: `t_nd = t / T` with `T = 31.62 s`
-- `context`:
  - Train: `[120, 7]`
  - Val/Test: `[20, 7]`
  - Fields: `[m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag]`
  - **How created**:
    - For each raw case, the generator stores dimensional parameters in `meta/params_used` (JSON dict).
    - Preprocessing (`process_raw_to_splits` in `src/data/preprocess.py`) scans all raw cases to find which keys actually exist and builds a canonical list from `CONTEXT_FIELDS`.
    - It then calls `build_context_vector(params_used, scales, fields=canonical_fields)` to:
      - Normalize each scalar with physics-aware rules (e.g. `m0 / M`, `Tmax / F`, `Isp / 250`, `Cd` as-is, `wind_mag / V`).
      - Pack them in a fixed order into a length‑7 vector.
    - All per‑case vectors are stacked into `inputs/context`, and the field names are saved as `meta/context_fields` so v1 and v2 agree on the ordering.
- `state`:
  - Train: `[120, 1501, 14]`
  - Val/Test: `[20, 1501, 14]`
  - Format: `[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]`

**Key Transformations (v1 and v2 share the same base preprocessing)**:
- **Nondimensionalization**: All state variables are normalized using physics-aware reference scales:
  - Length scale `L` (typically 10,000 m)
  - Velocity scale `V` (typically 313 m/s)
  - Time scale `T` (typically 31.62 s)
  - Mass scale `M` (typically 50 kg)
  - Force scale `F` (typically 490 N)
  - Angular velocity scale `W` (typically 0.0316 rad/s)
- **Context Normalization**: Context parameters normalized using physics-aware scaling
- **Consolidation**: Individual case files consolidated into train/val/test splits
- **Context Extraction**: Physical parameters extracted from metadata and assembled into context vectors

**Features (v1/v2)**:
- Time grid: `[n_cases, N]` where `N = 1501`
- Context vector: `[n_cases, 7]` (m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag)
- State trajectories: `[n_cases, N, 14]` (nondimensional)

**v2 Key Additions (on top of v1)**:
- **T_mag (Thrust Magnitude)**:
  - **Computation**: Extracted from control vector `u[:, 0]` (thrust component)
  - **Units**: Nondimensionalized using `F` scale: `T_mag_nd = T_mag / scales.F`
  - **Shape**: `[n_cases, N]`
  - **Purpose**: Provides time-varying thrust information directly to models

- **q_dyn (Dynamic Pressure)**:
  - **Computation**: `q_dyn = 0.5 * ρ(z) * |v|²`
    - Density: `ρ(z) = ρ₀ * exp(-z / h_scale)` (exponential atmosphere)
    - Speed: `|v| = sqrt(vx² + vy² + vz²)`
  - **Units**: Nondimensionalized using pressure scale: `q_dyn_nd = q_dyn / (F / L²)`
  - **Shape**: `[n_cases, N]`
  - **Purpose**: Provides aerodynamic loading information directly to models

**Backward Compatibility**: v2 is backward compatible with v1. All v1 functionality is preserved, and v2 adds optional features that models can use if supported. Direction D1.5.4 uses v2 data to leverage `T_mag` and `q_dyn` features via `InputBlockV2`.

---

### 1.3 Exploratory Data Analysis (EDA)

#### 1.3.1 State Variable Statistics

**Key Observations from Normalized State Data:**

| Variable | Mean | Std | Min | Max | Median | Notes |
|----------|------|-----|-----|-----|--------|-------|
| **z** (altitude) | 0.920 | 0.906 | 0.000 | 4.219 | 0.619 | Primary variable of interest |
| **vz** (vertical velocity) | 3.112 | 2.147 | 0.000 | 9.382 | 2.835 | High variance indicates diverse trajectories |
| **m** (mass) | 0.815 | 0.150 | 0.700 | 1.299 | 0.721 | Mass decreases over time (fuel consumption) |
| **x, y** (horizontal position) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Zero (vertical-only trajectories) |
| **vx, vy** (horizontal velocity) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Zero (vertical-only trajectories) |
| **q0, q1, q2, q3** (quaternion) | 1.0, 0.0, 0.0, 0.0 | 0.0 | - | - | - | Identity quaternion (no rotation) |
| **wx, wy, wz** (angular velocity) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Zero (no rotation) |

**Insights:**
- The dataset contains **vertical-only trajectories** (no horizontal motion)
- All cases start with **identity attitude** (no initial rotation)
- **Altitude (z)** shows high variance (std = 0.906), indicating diverse trajectory profiles
- **Vertical velocity (vz)** has the highest variance (std = 2.147), showing significant variation in ascent/descent rates
- **Mass** decreases from initial value (1.0) to minimum (0.7), consistent with fuel consumption

#### 1.3.2 Trajectory Characteristics

**Altitude Trajectories:**
- Trajectories show typical rocket ascent profiles: rapid initial climb, peak altitude, then descent
- Maximum normalized altitude: 4.219 (corresponds to ~42,190 m with L=10,000 m scale)
- Altitude distributions vary significantly over time, with maximum spread at mid-flight

**Vertical Velocity Trajectories:**
- Initial positive velocities (ascent phase)
- Velocity decreases as rocket approaches apogee
- Negative velocities during descent phase
- Maximum normalized velocity: 9.382 (corresponds to ~2,937 m/s with V=313 m/s scale)

**Mass Trajectories:**
- Monotonic decrease (fuel consumption)
- Initial mass: ~1.0 (normalized)
- Final mass: ~0.7 (normalized, 30% fuel consumed)
- Mass loss rate varies across cases based on thrust profiles

**Thrust Magnitude (T_mag):**
- Time-varying thrust profiles
- Typically high at launch, decreasing over time
- Provides direct control input information to the model

**Dynamic Pressure (q_dyn):**
- Increases with velocity during ascent
- Peaks at maximum velocity point
- Decreases with altitude (density decreases)
- Critical for aerodynamic force computation

#### 1.3.3 Context Parameter Analysis

**Context Parameter Distributions:**
- All 7 context parameters are sampled from their respective parameter spaces
- Distributions show good coverage of the parameter space (LHS/Sobol sampling)
- No strong correlations between most parameters (independent sampling)

**Context Parameter Ranges (Normalized):**
- Parameters are normalized using physics-aware scaling
- Distributions ensure diverse trajectory characteristics
- Each parameter affects different aspects of the dynamics:
  - `m0`: Initial mass → affects total fuel capacity
  - `Isp`: Specific impulse → affects fuel efficiency
  - `Cd`: Drag coefficient → affects aerodynamic drag
  - `CL_alpha`, `Cm_alpha`: Aerodynamic coefficients → affect lift and moments
  - `Tmax`: Maximum thrust → affects acceleration capability
  - `wind_mag`: Wind magnitude → affects external forces

**Max Altitude vs Context Parameters:**
- Maximum altitude correlates with:
  - **Tmax**: Higher thrust → higher altitude
  - **m0**: More fuel → higher altitude
  - **Isp**: Better efficiency → higher altitude
- Negative correlation with:
  - **Cd**: Higher drag → lower altitude
  - **wind_mag**: Higher wind → lower altitude (energy loss)

#### 1.3.4 Data Quality Observations

**Strengths:**
-  Diverse trajectory profiles (high variance in z, vz)
-  Consistent time resolution (1501 points per trajectory)
-  Physics-consistent data (generated from OCP solver)
-  Good parameter space coverage (LHS/Sobol sampling)

**Characteristics:**
- Vertical-only trajectories (simplified 1D motion)
- No initial rotation (identity quaternion)
- Monotonic mass decrease (realistic fuel consumption)
- Smooth trajectories (no discontinuities)

**Implications for Modeling:**
- Model can focus on vertical dynamics (simplified problem)
- No need to handle complex rotational dynamics in this dataset
- Mass prediction is critical (affects all other predictions)
- Altitude and vertical velocity are the primary outputs of interest

---

### 1.4 Model Inputs and Outputs

#### Context Variables (Case-level Inputs)

The model receives a **7-dimensional context vector** per case (constant across time):

| Index | Variable | Symbol | Description | Range |
|-------|----------|--------|-------------|-------|
| 0 | Initial Mass | `m0` | Initial rocket mass (kg) | `[45, 65]` |
| 1 | Specific Impulse | `Isp` | Propellant efficiency (s) | `[220, 280]` |
| 2 | Drag Coefficient | `Cd` | Drag coefficient (-) | `[0.25, 0.45]` |
| 3 | Lift Curve Slope | `CL_alpha` | Lift coefficient per unit angle of attack (1/rad) | `[2.5, 4.5]` |
| 4 | Pitch Moment Coefficient | `Cm_alpha` | Pitch moment coefficient per unit angle of attack (1/rad) | `[-1.2, -0.4]` |
| 5 | Maximum Thrust | `Tmax` | Maximum available thrust (N) | `[3000, 5000]` |
| 6 | Wind Magnitude | `wind_mag` | Wind speed magnitude (m/s) | `[0, 15]` |

- **Tensor Shape**: `[n_cases, 7]`
- **Normalization**: Each field uses physics-aware scaling (mass by `M`, thrust by `F`, velocities by `V`, etc.).

#### State Variables (Trajectory Targets)

The model predicts a **14-dimensional state vector** at each time step:

| Index | Variable | Symbol | Description |
|-------|----------|--------|-------------|
| 0-2 | Position | `[x, y, z]` | 3D position coordinates (m, nondimensional) |
| 3-5 | Velocity | `[vx, vy, vz]` | 3D velocity components (m/s, nondimensional) |
| 6-9 | Quaternion | `[q0, q1, q2, q3]` | Attitude quaternion (unit quaternion) |
| 10-12 | Angular Velocity | `[wx, wy, wz]` | Angular velocity components (rad/s, nondimensional) |
| 13 | Mass | `m` | Rocket mass (kg, nondimensional) |

- **Tensor Shape**: `[n_cases, N, 14]` where `N = 1501`
- **Normalization**: Positions by `L`, velocities by `V`, time derivatives by `T`, mass by `M`, angular rates by `W`.

#### Additional Time-Series Features (v2-only)

| Feature | Symbol | Description | Shape |
|---------|--------|-------------|-------|
| Thrust Magnitude | `T_mag` | Time-varying thrust magnitude (nondimensional) | `[n_cases, N]` |
| Dynamic Pressure | `q_dyn` | Time-varying dynamic pressure (nondimensional) | `[n_cases, N]` |

Computation:
- `T_mag`: Extracted from control vector `u[:, 0]`, scaled by `F = 490 N`.
- `q_dyn`: `0.5 * ρ(z) * |v|²`, scaled by `F / L²` (`ρ(z) = ρ₀ * exp(-z / h_scale)`).

**Why this matters (short explanation).** Altitude `z` and vertical velocity `vz` remain the critical targets because they directly encode mission success (reaching target apogee), safety (dynamic pressure), and performance metrics. Mass, quaternion, and angular velocity are still predicted to preserve coupled dynamics (mass → attitude → translation) even though they are less critical for evaluation.

#### Batch Usage

Direction D1.5.4 consumes the following tensors per batch:
- `t`: `[batch, N, 1]`
- `context`: `[batch, 7]`
- `T_mag`: `[batch, N, 1]` (optional for v1 data, zeros if absent)
- `q_dyn`: `[batch, N, 1]` (optional for v1 data)

The network outputs `state`: `[batch, N, 14]`, representing the full trajectory for each sample.

---

## 2. Loss Function

### 2.1 Total Loss Structure

**Total Loss Structure:**
```
L_total = λ_data·L_data + λ_phys·L_phys + λ_bc·L_bc + L_soft_physics
```

### 2.2 Component Losses

1. **Data Loss** (`L_data`, `λ_data = 1.0`):
   - Component-weighted MSE between predicted and true states
   - Separate weights for translation, rotation, and mass
   - Mass component weight: `4.0` (higher importance)

2. **Physics Loss** (`L_phys`, `λ_phys = 0.1 → 1.0`):
   - ODE residual computed via automatic differentiation
   - Enforces: `∂state/∂t = f(state, control, params)`
   - Uses 6-DOF rocket dynamics function

3. **Boundary Loss** (`L_bc`, `λ_bc = 1.0`):
   - Initial condition enforcement
   - `L_bc = ||state_pred(t=0) - state_true(t=0)||²`

4. **Soft Physics Residuals** (Direction D1.5):
   - **Mass Residual** (`λ_mass_residual = 0.025`): Enforces mass flow rate consistency
   - **Vertical Velocity Residual** (`λ_vz_residual = 0.025`): Enforces vertical acceleration consistency
   - **Horizontal Velocity Residual** (`λ_vxy_residual = 0.005`): Enforces horizontal acceleration consistency
   - **Smoothing Losses**:
     - `λ_smooth_z = 5.0e-5`: Smooths altitude trajectory
     - `λ_smooth_vz = 1.0e-5`: Smooths vertical velocity trajectory

5. **Position-Velocity Consistency** (Direction D1.51):
   - **Position-Velocity Loss** (`λ_pos_vel = 0.5`): Enforces `p(t+1) ≈ p(t) + v(t)·Δt`

### 2.3 Loss Weight Scheduling

**Two-Phase Training**: 
- Phase 1 (0-55% epochs): Soft physics losses disabled (`λ = 0`)
- Phase 2 (55-100% epochs): Soft physics losses ramped up (cosine schedule)

**Homotopy Schedule**: Physics loss weight increases from `0.1` to `1.0` over training

### 2.4 How Loss Function Works

1. **Forward Pass**: Model predicts state trajectory
2. **Data Loss**: Compare predicted vs. true states (supervised learning)
3. **Physics Loss**: 
   - Compute `∂state_pred/∂t` via autograd
   - Evaluate ODE residual: `r = ∂state/∂t - f(state, control, params)`
   - Penalize residual: `L_phys = mean(r²)`
4. **Boundary Loss**: Enforce initial conditions
5. **Soft Physics**: Additional constraints for smoothness and consistency
6. **Total Loss**: Weighted sum of all components
7. **Backward Pass**: Gradient flows through all loss components

### 2.5 Training Strategy

- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: Cosine annealing (T_max=160, eta_min=1e-6)
- **Gradient Clipping**: Max norm = 1.0
- **Early Stopping**: Patience = 25 (Phase 1), 40 (Phase 2)

---

## 3. Model Architecture

### 3.1 Model Overview

**Model Type**: `DirectionDPINN_D154` (Direction D1.5.4)

**Architecture Overview:**

```mermaid
flowchart LR
    subgraph Inputs
        T[Time t]
        C[Context (m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag)]
        TM[T_mag (v2)]
        QD[q_dyn (v2)]
    end

    subgraph InputBlockV2
        TE[Time Embedding\n(Fourier)]
        CE[Context Encoder\nMLP]
        EE[Extra Features\nMLP (T_mag, q_dyn)]
    end

    subgraph Backbone
        B1[MLP Layer 1]
        B2[MLP Layer 2]
        B3[MLP Layer 3]
        B4[MLP Layer 4]
    end

    subgraph Heads
        G3[Head G3:\nMass m]
        G2[Head G2:\nAttitude q,\nAngular vel w]
        G1[Head G1:\nPosition x,\nVelocity v]
    end

    subgraph Output
        S[State Trajectory\n(x, y, z, vx, vy, vz,\nq0..q3, wx, wy, wz, m)]
    end

    %% Input fusion
    T --> TE
    C --> CE
    TM --> EE
    QD --> EE

    TE -->|concat| B1
    CE -->|concat| B1
    EE -->|concat| B1

    %% Backbone
    B1 --> B2 --> B3 --> B4

    %% Dependency-aware heads
    B4 --> G3
    B4 --> G2
    B4 --> G1

    G3 -->|m| G2
    G2 -->|q, w| G1

    %% Final state assembly
    G1 --> S
    G2 --> S
    G3 --> S
```

### 3.2 Detailed Structure

1. **InputBlockV2** (Feature Fusion):
   - **Time Embedding**: Fourier features (8 frequencies → 17D)
     - `[t, sin(2πk t), cos(2πk t)]` for k=1..8
   - **Context Encoder**: MLP (7 → 32D)
   - **Extra Features Embedding**: MLP (2 → 16D) for `[T_mag, q_dyn]`
   - **Output**: Fused features `[batch, N, 65D]` (17 + 32 + 16)

2. **Shared Backbone**:
   - **Architecture**: MLP with 4 hidden layers
   - **Hidden Dimensions**: `[256, 256, 256, 256]`
   - **Activation**: GELU
   - **Output**: Latent representation `[batch, N, 256D]`

3. **Head G3 (Mass Prediction)**:
   - **Input**: Latent `[256D]`
   - **Architecture**: MLP `[256 → 128 → 64 → 1]`
   - **Output**: Mass `m` `[1D]`
   - **Special**: Optional mass monotonicity enforcement via softplus deltas

4. **Head G2 (Attitude + Angular Velocity)**:
   - **Input**: Concatenated `[latent (256D) + m (1D)] = [257D]`
   - **Architecture**: MLP `[257 → 256 → 128 → 64 → 9D]`
   - **Output**: 
     - 6D rotation representation `[6D]` (if `use_rotation_6d=True`)
     - Angular velocity `[wx, wy, wz]` `[3D]`
   - **Special**: Converts 6D rotation to quaternion for output

5. **Head G1 (Translation)**:
   - **Input**: Concatenated `[latent (256D) + m (1D) + q (4D) + w (3D)] = [264D]`
   - **Architecture**: MLP `[264 → 256 → 128 → 128 → 64 → 6D]`
   - **Output**: Position and velocity `[x, y, z, vx, vy, vz]` `[6D]`

**Dependency Chain**: G3 (mass) → G2 (attitude) → G1 (translation)

**Key Features:**
- **6D Rotation Representation**: Stable rotation encoding (avoids quaternion normalization issues)
- **Physics-Aware Zero-Aero Handling**: Automatically sets attitude to identity for zero-aerodynamics cases
- **Horizontal Motion Suppression**: Zeroes horizontal position/velocity for zero-aero cases

### 3.3 How the Model Works

**Forward Pass Flow:**

1. **Input Processing**:
   - Time `t` is embedded via Fourier features
   - Context `context` is encoded via MLP
   - `T_mag` and `q_dyn` are embedded via small MLP
   - All features are concatenated and passed to backbone

2. **Feature Extraction**:
   - Backbone processes fused features to extract general motion information
   - Produces latent representation shared across all heads

3. **Sequential Prediction**:
   - **Step 1**: Head G3 predicts mass `m` from latent
   - **Step 2**: Head G2 predicts attitude `q` and angular velocity `w` from `[latent, m]`
   - **Step 3**: Head G1 predicts position `x` and velocity `v` from `[latent, m, q, w]`

4. **Output Assembly**:
   - All predictions are concatenated: `[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]`
   - Special handling for zero-aerodynamics cases (identity quaternion, zero horizontal motion)

**Why This Architecture Works:**
- **Dependency Preservation**: Explicit dependency chain matches physical relationships
- **Feature Reuse**: Shared backbone extracts common motion patterns
- **Physics Integration**: v2 features (`T_mag`, `q_dyn`) provide direct physics information
- **Stable Representations**: 6D rotation avoids quaternion normalization issues
