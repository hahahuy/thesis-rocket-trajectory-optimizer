# Rocket Dynamics Library Design

## Overview

This document describes the design and architecture of the 6-degree-of-freedom rocket dynamics library with atmospheric effects, wind models, and structural constraints.

## State Representation

### State Vector (14 elements)

The state vector is standardized as:

```
[0-2]:   x, y, z          Position in inertial frame [m]
[3-5]:   vx, vy, vz       Velocity in inertial frame [m/s]
[6-9]:   q0, q1, q2, q3   Quaternion (body to inertial) [dimensionless]
[10-12]: wx, wy, wz       Angular velocity in body frame [rad/s]
[13]:    m                Mass [kg]
```

**C++ Structure:**
```cpp
struct State {
    Vec3 r_i;           // Position [m]
    Vec3 v_i;           // Velocity [m/s]
    Quaterniond q_bi;   // Quaternion from body to inertial
    Vec3 w_b;           // Angular velocity [rad/s]
    double m;           // Mass [kg]
};
```

**Index Enums:**
```cpp
enum class StateIndex {
    X = 0, Y, Z, VX, VY, VZ, Q0, Q1, Q2, Q3, WX, WY, WZ, M
};
```

## Control Representation

### Control Vector (5 elements)

```
[0]:     T                Thrust magnitude [N]
[1-3]:   uT_x, uT_y, uT_z Thrust direction unit vector in body frame
[4]:     delta            Control surface deflection [rad]
```

**C++ Structure:**
```cpp
struct Control {
    double T;        // Thrust magnitude [N]
    Vec3 uT_b;       // Thrust direction unit vector
    double delta;    // Control surface deflection [rad]
};
```

**Index Enums:**
```cpp
enum class ControlIndex {
    T = 0, THETA, PHI, UT_Z, DELTA
};
```

## Key Equations

### Dynamic Pressure
```
q = 0.5 * ρ * V²
```

### Normal Load Factor
```
n = |L| / (m * g₀) + 1.0
L = q * S_ref * CL_alpha * α
```

### Pitch Moment (with control surfaces)
```
M_pitch = q * S_ref * l_ref * (Cm_alpha * α + C_delta * δ)
```

### Quaternion Derivative
```
q̇ = 0.5 * q * [0, ω]
```

### State Derivatives
```
ṙ = v
v̇ = F / m
q̇ = 0.5 * q * [0, ω]
ω̇ = I⁻¹ * (M - ω × Iω)
ṁ = -T / (Isp * g₀)
```

## Physical Parameters

### Aerodynamic
- `Cd = 0.3` - Drag coefficient
- `CL_alpha = 3.5 [1/rad]` - Lift curve slope
- `Cm_alpha = -0.8 [1/rad]` - Pitch moment coefficient
- `C_delta = 0.05 [1/rad]` - Control surface authority
- `S_ref = 0.05 [m²]` - Reference area
- `l_ref = 1.2 [m]` - Reference length

### Propulsion
- `Isp = 300 [s]` - Specific impulse
- `g₀ = 9.81 [m/s²]` - Standard gravity

### Atmospheric
- `ρ₀ = 1.225 [kg/m³]` - Sea level density
- `h_scale = 8400 [m]` - Atmospheric scale height

## Constraints

### Dynamic Pressure Limit
- `q_max = 40000 [Pa]` - Maximum dynamic pressure

### Normal Load Factor Limit
- `n_max = 5.0 [g]` - Maximum load factor

### Thrust Limit
- `T_max = 4000 [N]` - Maximum thrust (nominal case)

### Mass Limit
- `m_dry = 10 [kg]` - Dry mass

## Numerical Integration

### RK4 (Fixed Step)
- 4th-order Runge-Kutta
- Suitable for real-time simulation
- Constant time step

### RK45 (Adaptive Step)
- Embedded 4th/5th-order Runge-Kutta
- Automatic step size adjustment
- Error tolerance: 1e-6 (default)
- Minimum step: 1e-9
- Maximum step: 1.0

## Scaling

### Reference Scales (for O(1) normalization)

```
L_ref = 1e4 [m]      (10 km)
V_ref = 1e3 [m/s]    (1000 m/s)
M_ref = 50 [kg]
F_ref = 5e3 [N]      (5000 N)
T_ref = 50 [s]
Q_ref = 1e4 [Pa]     (10 kPa)
```

**Non-dimensionalization:**
```cpp
state_nd = nondimensionalize(state, scales)
```

**Dimensionalization:**
```cpp
state = dimensionalize(state_nd, scales)
```

## Smooth Functions

For automatic differentiation compatibility, all non-smooth functions use smooth approximations:

- `smooth_max(a, b)` - Smooth maximum using softplus
- `smooth_min(a, b)` - Smooth minimum
- `smooth_clamp(x, lo, hi)` - Smooth clamping
- `smooth_atan2(y, x)` - Smooth atan2 (avoids NaN at origin)

## File Structure

```
src/physics/
├── types.hpp          - State, Control, Phys, Limits, Diag structures
├── dynamics.hpp/cpp   - 6-DOF dynamics equations
├── integrator.hpp/cpp - RK4 and RK45 integrators
├── atmosphere.hpp/cpp - Atmospheric models (ISA, exponential)
├── constraints.hpp/cpp - Constraint checking and handling
├── smooth_functions.hpp - Smooth approximations for AD
└── utils.hpp         - Utility functions

src/utils/
├── scaling.hpp/cpp    - Non-dimensionalization utilities

configs/
├── phys.yaml          - Physical parameters
├── scales.yaml        - Reference scales
└── limits.yaml        - Operational limits

tests/
├── test_types.cpp     - Type structure tests
├── test_dynamics.cpp  - Dynamics tests
└── test_constraints.cpp - Constraint tests

bin/
├── run_types_demo.cpp      - Types demonstration
├── run_dynamics_demo.cpp   - Dynamics demonstration
└── validate_dynamics.cpp   - Pre-WP2 validation
```

## Integration with OCP (WP2)

The physics library is designed for seamless integration with optimal control:

1. **Path Constraints:**
   - `q(t) ≤ q_max`
   - `n(t) ≤ n_max`
   - `m(t) ≥ m_dry`

2. **State Bounds:**
   - Altitude: `h ≥ 0`
   - Mass: `m ≥ m_dry`

3. **Control Bounds:**
   - Thrust: `0 ≤ T ≤ T_max`
   - Gimbal: Unit vector constraint `||uT|| = 1`
   - Control surface: `|δ| ≤ δ_max`

## Integration with PINN (WP4)

The library supports Physics-Informed Neural Networks:

1. **Differentiability:**
   - All functions use smooth approximations
   - Compatible with automatic differentiation

2. **Physics Residuals:**
   - RHS function `f(x, u, t)` is differentiable
   - Can be used in PINN loss function

3. **Constraint Penalties:**
   - Soft penalty terms for q and n constraints
   - Differentiable penalty functions

## Validation Status

See `docs/pre_wp2_validation.md` for detailed validation status.

**Current Status:** ~70% complete
- ✅ Stability and continuity verified
- ✅ Scaling implemented
- ✅ Smooth functions integrated
- ✅ State/control order standardized
- 🔄 Reference flight case pending execution
- 🔄 Constraint validation pending plots
- 🔄 Logging pending implementation
- 🔄 Documentation finalization pending

## References

- WP1 comprehensive: `docs/wp1_comprehensive_description.md`
- WP2 comprehensive: `docs/wp2_comprehensive_description.md`
- CI Setup: `docs/ci_setup.md`
