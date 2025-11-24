# Rocket Trajectory Optimizer - Design Overview

This document provides a high-level overview of the system architecture and design. For detailed implementation information, see the comprehensive work package descriptions referenced inline below.

## System Architecture

The Rocket Trajectory Optimizer is organized into four main work packages (WPs):

1. **WP1 - Physics Core**: 6-DOF rocket dynamics library in C++
2. **WP2 - Optimal Control Baseline**: Direct collocation solver using CasADi/IPOPT
3. **WP3 - Dataset Generation**: Parameter space sweeping and preprocessing pipeline
4. **WP4 - PINN Training**: Physics-Informed Neural Network models and training

For detailed architecture information, see:
- **WP1**: [wp1_comprehensive_description.md](wp1_comprehensive_description.md#implementation-architecture)
- **WP2**: [wp2_comprehensive_description.md](wp2_comprehensive_description.md#implementation-architecture)
- **WP3**: [wp3_comprehensive_description.md](wp3_comprehensive_description.md#implementation-architecture)
- **WP4**: [wp4_comprehensive_description.md](wp4_comprehensive_description.md#implementation-details)

## State and Control Representations

### State Vector (14 elements)

The state vector is standardized across all work packages:

```
[0-2]:   x, y, z          Position in inertial frame [m]
[3-5]:   vx, vy, vz       Velocity in inertial frame [m/s]
[6-9]:   q0, q1, q2, q3   Quaternion (body to inertial) [dimensionless]
[10-12]: wx, wy, wz       Angular velocity in body frame [rad/s]
[13]:    m                Mass [kg]
```

**C++ Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#state-and-control-representations) for the `State` structure in `src/physics/types.hpp` with Eigen support and validation utilities.

**Python/CasADi Implementation**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#state-and-control-variables) for how states are handled in the OCP solver.

### Control Vector (5 elements)

```
[0]:     T                Thrust magnitude [N]
[1-3]:   uT_x, uT_y, uT_z Thrust direction unit vector in body frame
[4]:     delta            Control surface deflection [rad]
```

**C++ Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#state-and-control-representations) for the `Control` structure.

**OCP Solver**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#state-and-control-variables) for control parameterization in the collocation solver.

## Physics and Dynamics

### 6-DOF Dynamics Equations

The core dynamics are implemented in WP1 and used by WP2 and WP4:

- **Position**: `ṙ = v`
- **Velocity**: `v̇ = (F_thrust + F_drag + F_lift)/m + g`
- **Quaternion**: `q̇ = 0.5 * q ⊗ [0, ω]`
- **Angular velocity**: `ω̇ = I⁻¹ (M - ω × Iω)`
- **Mass**: `ṁ = -T / (Isp * g₀)`

**Detailed Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#dynamics-6-dof) for the complete dynamics implementation in `src/physics/dynamics.cpp`, including aerodynamics, atmosphere models, and moment calculations.

**CasADi Symbolic Version**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#problem-formulation) for how dynamics are represented symbolically for automatic differentiation in the OCP solver.

**PyTorch Version**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#2-model-architecture) for the PyTorch implementation used in PINN physics loss computation.

### Key Physical Parameters

**Aerodynamic**:
- `Cd = 0.3` - Drag coefficient
- `CL_alpha = 3.5 [1/rad]` - Lift curve slope
- `Cm_alpha = -0.8 [1/rad]` - Pitch moment coefficient
- `C_delta = 0.05 [1/rad]` - Control surface authority
- `S_ref = 0.05 [m²]` - Reference area
- `l_ref = 1.2 [m]` - Reference length

**Propulsion**:
- `Isp = 300 [s]` - Specific impulse
- `g₀ = 9.81 [m/s²]` - Standard gravity

**Atmospheric**:
- `ρ₀ = 1.225 [kg/m³]` - Sea level density
- `h_scale = 8400 [m]` - Atmospheric scale height

**Configuration**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#implementation-architecture) for where these parameters are defined (`configs/phys.yaml`).

## Constraints

### Path Constraints

1. **Dynamic Pressure**: `q = 0.5 * ρ * V²` with `q_max = 40000 [Pa]`
2. **Normal Load Factor**: `n = |L|/(m g₀) + 1` with `n_max = 5.0 [g]`
3. **Mass Floor**: `m ≥ m_dry = 10 [kg]`

**WP1 Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#constraints-and-diagnostics) for constraint checking and diagnostics in `src/physics/constraints.cpp`.

**WP2 Enforcement**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#problem-formulation) for how path constraints are enforced in the OCP solver via CasADi.

**WP3 Validation**: See [wp3_comprehensive_description.md](wp3_comprehensive_description.md#quality-gates) for constraint validation in the dataset generation pipeline.

### State and Control Bounds

- **Altitude**: `h ≥ 0`
- **Mass**: `m ≥ m_dry`
- **Thrust**: `0 ≤ T ≤ T_max`
- **Gimbal**: Unit vector constraint `||uT|| = 1`
- **Control surface**: `|δ| ≤ δ_max`

**Detailed Bounds**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#problem-formulation) for complete bounds specification in the OCP.

## Numerical Integration

### Integration Methods

1. **RK4 (Fixed Step)**: 4th-order Runge-Kutta, suitable for real-time simulation
2. **RK45 (Adaptive Step)**: Embedded 4th/5th-order Runge-Kutta with automatic step size adjustment

**WP1 Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#numerical-integration) for integrator details in `src/physics/integrator.hpp/cpp`.

**WP3 Usage**: See [wp3_comprehensive_description.md](wp3_comprehensive_description.md#technical-overview) for how integration is used in dataset generation.

## Scaling and Normalization

### Reference Scales (for O(1) normalization)

```
L_ref = 1e4 [m]      (10 km)
V_ref = 1e3 [m/s]    (1000 m/s)
M_ref = 50 [kg]
F_ref = 5e3 [N]      (5000 N)
T_ref = 50 [s]
Q_ref = 1e4 [Pa]     (10 kPa)
```

**WP1 Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#scaling) for non-dimensionalization utilities in `src/utils/scaling.hpp/cpp`.

**WP2 Usage**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#scaling-and-normalization) for how scaling is applied in the OCP solver.

**WP3 Preprocessing**: See [wp3_comprehensive_description.md](wp3_comprehensive_description.md#technical-overview) for scaling in dataset preprocessing.

**WP4 Training**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#1-data-loading-and-preprocessing) for normalization in PINN training.

## Smooth Functions

For automatic differentiation compatibility, all non-smooth functions use smooth approximations:

- `smooth_max(a, b)` - Smooth maximum using softplus
- `smooth_min(a, b)` - Smooth minimum
- `smooth_clamp(x, lo, hi)` - Smooth clamping
- `smooth_atan2(y, x)` - Smooth atan2 (avoids NaN at origin)

**Implementation**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#implementation-architecture) for smooth functions in `src/physics/smooth_functions.hpp`.

**Usage in WP2**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#automatic-differentiation-issues) for how smooth functions ensure AD compatibility in CasADi.

## File Structure

### WP1 - Physics Core

```
src/physics/
├── types.hpp              # State, Control, Phys, Limits, Diag structures
├── dynamics.hpp/cpp       # 6-DOF dynamics equations
├── integrator.hpp/cpp     # RK4 and RK45 integrators
├── atmosphere.hpp/cpp     # Atmospheric models (ISA, exponential)
├── constraints.hpp/cpp    # Constraint checking and handling
├── smooth_functions.hpp   # Smooth approximations for AD
└── utils.hpp             # Utility functions

src/utils/
├── scaling.hpp/cpp       # Non-dimensionalization utilities
```

**Detailed Structure**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#implementation-architecture).

### WP2 - Optimal Control Solver

```
src/solver/
├── dynamics_casadi.py    # CasADi symbolic dynamics (6-DOF)
├── collocation.py        # Hermite-Simpson collocation
├── constraints.py        # Path and boundary constraints
└── utils.py             # Initial guess, scaling helpers
```

**Detailed Structure**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#implementation-architecture).

### WP3 - Dataset Generation

```
src/data/
├── generator.py          # Parameter sweeping, OCP solving
├── preprocess.py         # Normalization, splitting
└── storage.py            # HDF5 I/O, dataset cards
```

**Detailed Structure**: See [wp3_comprehensive_description.md](wp3_comprehensive_description.md#implementation-architecture).

### WP4 - PINN Models

```
src/models/
├── pinn.py               # Base PINN model
├── residual_net.py       # Hybrid residual model
├── architectures.py      # MLP blocks, embeddings
├── latent_ode.py         # Direction A: Latent ODE
├── sequence_pinn.py      # Direction B: Transformer
└── hybrid_pinn.py        # Direction C, C1, C2: Hybrid models

src/train/
├── train_pinn.py         # Training loop
├── losses.py             # Physics/data/boundary losses
└── callbacks.py          # Schedulers, early stopping
```

**Detailed Structure**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#structure-overview).

**PINN Architecture Evolution**: See [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md) for the complete history of PINN model development from baseline through Directions A, B, C, C1, and C2.

## Integration Between Work Packages

### WP1 → WP2 Integration

The physics library (WP1) is integrated with the OCP solver (WP2):

1. **Path Constraints**: `q(t) ≤ q_max`, `n(t) ≤ n_max`, `m(t) ≥ m_dry`
2. **State Bounds**: Altitude `h ≥ 0`, mass `m ≥ m_dry`
3. **Control Bounds**: Thrust, gimbal, control surface limits

**Details**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#integration-with-wp1) for how WP1 physics are used in WP2.

### WP2 → WP3 Integration

The OCP solver (WP2) generates reference trajectories for dataset creation (WP3):

1. **OCP Solving**: Parameter space sweeping with WP2 solver
2. **Trajectory Integration**: Using WP1 integrators to standard time grids
3. **Quality Validation**: Constraint checking and feasibility verification

**Details**: See [wp3_comprehensive_description.md](wp3_comprehensive_description.md#technical-overview) for the dataset generation pipeline.

### WP3 → WP4 Integration

Processed datasets (WP3) are used for PINN training (WP4):

1. **Data Loading**: HDF5 datasets with normalized inputs/targets
2. **Context Vectors**: Physics-aware normalization parameters
3. **Time Grids**: Standardized time discretization

**Details**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#1-data-loading-and-preprocessing) for data loading in PINN training.

### WP1 → WP4 Integration

The physics library (WP1) provides differentiable dynamics for PINN physics loss:

1. **Differentiability**: All functions use smooth approximations
2. **Physics Residuals**: RHS function `f(x, u, t)` is differentiable
3. **Constraint Penalties**: Soft penalty terms for q and n constraints

**Details**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#2-model-architecture) for how WP1 physics are used in PINN loss computation.

## PINN Architecture Evolution

The PINN models have evolved through multiple architectural directions:

1. **Baseline PINN**: Vanilla MLP with Fourier features (see [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md#2025-11-15-original-baseline-pinn-model))
2. **Direction A**: Latent Neural ODE PINN (see [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md#2025-11-18-direction-a--latent-ode-implementation))
3. **Direction B**: Sequence Transformer PINN (see [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md#2025-11-18-direction-b--sequence-transformer-pinn))
4. **Direction C**: Hybrid Sequence + Latent ODE (see [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md#2025-11-18-direction-c--hybrid-sequence--latent-ode-pinn))
5. **Direction C1**: Enhanced Hybrid with stability improvements (see [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md#2025-11-19-direction-c1--hybrid-stability--context-upgrades))
6. **Direction C2**: Shared Stem + Dedicated Branches (see [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md#2025-11-19-direction-c2--shared-stem--dedicated-branches))

**Architecture Diagrams**: See [architecture_diagram.md](architecture_diagram.md) for Mermaid diagrams of all architectures.

**Research Notes**: See [thesis_notes.md](thesis_notes.md) for detailed research notes and development decisions.

## Validation and Testing

### WP1 Validation

- Unit tests for dynamics, constraints, integrators
- Pre-WP2 validation executable
- Quaternion normalization verification

**Details**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#testing--validation-framework).

### WP2 Validation

- 26+ unit tests with ≥88% coverage
- Integration tests with WP1 comparison
- Robustness sweeps (100% convergence)

**Details**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#testing--validation-framework).

### WP3 Validation

- Quality gates: 90%+ success rate
- Constraint checking (0 violations)
- Quaternion normalization (max error 1e-07)

**Details**: See [wp3_comprehensive_description.md](wp3_comprehensive_description.md#testing--validation-framework).

### WP4 Validation

- RMSE metrics per component
- Physics residual analysis
- Quaternion norm tracking
- Mass monotonicity checks

**Details**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#evaluation-and-metrics).

**Cross-WP Validation**: See [RESULTS_AND_VALIDATION.md](RESULTS_AND_VALIDATION.md) for comprehensive validation results across all work packages and experiment summaries.

## Configuration Files

### Physics Configuration

- `configs/phys.yaml`: Physical parameters (aerodynamics, propulsion, atmosphere)
- `configs/scales.yaml`: Reference scales for normalization
- `configs/limits.yaml`: Operational limits (q_max, n_max, T_max, etc.)

**Details**: See [wp1_comprehensive_description.md](wp1_comprehensive_description.md#implementation-architecture).

### OCP Configuration

- `configs/ocp.yaml`: OCP solver settings (collocation points, solver options, linear solver)

**Details**: See [wp2_comprehensive_description.md](wp2_comprehensive_description.md#configuration).

### Training Configuration

- `configs/train.yaml`: Training hyperparameters
- `configs/model_*.yaml`: Model architecture configurations

**Details**: See [wp4_comprehensive_description.md](wp4_comprehensive_description.md#configuration).

## References

- **WP1 Comprehensive**: [wp1_comprehensive_description.md](wp1_comprehensive_description.md)
- **WP2 Comprehensive**: [wp2_comprehensive_description.md](wp2_comprehensive_description.md)
- **WP3 Comprehensive**: [wp3_comprehensive_description.md](wp3_comprehensive_description.md)
- **WP4 Comprehensive**: [wp4_comprehensive_description.md](wp4_comprehensive_description.md)
- **Architecture Changelog**: [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md)
- **Architecture Diagrams**: [architecture_diagram.md](architecture_diagram.md)
- **Results & Validation**: [RESULTS_AND_VALIDATION.md](RESULTS_AND_VALIDATION.md)
- **Setup Guide**: [SETUP.md](SETUP.md)
- **Documentation Structure**: [docs/README.md](README.md)

