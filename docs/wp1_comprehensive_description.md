# WP1 - Physics Core: Comprehensive Description

**Last Updated**: 2025-11-03  
**Status**: ✅ Core physics implemented, validated, and integrated with constraints

---

## Table of Contents

1. Executive Summary
2. Technical Overview
3. Implementation Architecture
4. Development Journey
5. Testing & Validation Framework
6. Current Status & Results
7. Lessons Learned
8. Future Recommendations
9. References
10. WP1 Operations: How to Run

---

## Executive Summary

WP1 establishes a robust, strongly-typed 6-DOF rocket physics core in C++ with atmospheric effects, aerodynamic forces/moments, structural constraints (dynamic pressure q, load factor n), and diagnostics. It serves as the validated physical foundation for WP2 (OCP) and later PINN work.

### Key Achievements

- ✅ Strongly typed state/control/parameter structures with Eigen support
- ✅ 6-DOF dynamics with aerodynamics, gravity, propulsion, wind hooks
- ✅ Structural constraints and diagnostics (q, n, α, m underflow)
- ✅ Smooth functions for AD-compatibility and numerical stability
- ✅ Comprehensive unit tests and demos; pre-WP2 validation executable

---

## Technical Overview

### State and Control Representations

State (14): `[x,y,z, vx,vy,vz, q0,q1,q2,q3, wx,wy,wz, m]`  
Control (5): `T, uT_b(x,y,z), δ`

C++ structures (see `src/physics/types.hpp`) provide:
- `State`, `Control`, `Phys`, `Limits`, `Diag`
- Eigen vector conversions (round-trip), validation utilities

### Dynamics (6-DOF)

- Position: `ṙ = v`
- Velocity: `v̇ = (F_thrust + F_drag + F_lift)/m + g`
- Quaternion: `q̇ = 0.5 * q ⊗ [0, ω]`
- Angular velocity: `ω̇ = I⁻¹ (M - ω × Iω)`
- Mass: `ṁ = -T / (Isp * g₀)`

Aerodynamics and atmosphere (see `src/physics/dynamics.cpp`):
- Exponential density model; drag and lift with AoA
- Moments include control-surface effects and lever arms

### Constraints and Diagnostics

- Dynamic pressure: `q = 0.5 ρ V²` with `q_max` limit
- Load factor: `n = |L|/(m g₀) + 1` with `n_max` limit
- Mass floor: `m ≥ m_dry`
- Diagnostic flags: `q_violation`, `n_violation`, `m_underflow`, `alpha`

### Numerical Integration

- RK4 (fixed step) and RK45 (adaptive step) integrators
- Smooth functions used to avoid non-differentiabilities

### Scaling

Reference scales (see `configs/scales.yaml`) for O(1) normalization:  
`L_ref=1e4, V_ref=1e3, M_ref=50, F_ref=5e3, T_ref=50, Q_ref=1e4`  
Utilities in `src/utils/scaling.hpp/cpp` for non/di-mensionalization.

---

## Implementation Architecture

```
src/physics/
  types.hpp              # State, Control, Phys, Limits, Diag
  dynamics.hpp/cpp       # 6-DOF dynamics + aero/moments
  constraints.hpp/cpp    # q/n checks, enforce limits, diagnostics
  integrator.hpp/cpp     # RK4, RK45
  atmosphere.hpp/cpp     # Atmospheric models
  smooth_functions.hpp   # Smooth max/min/clamp/atan2

src/utils/
  scaling.hpp/cpp        # Non-dimensionalization utilities

tests/
  test_types.cpp         # Types tests
  test_dynamics.cpp      # Dynamics tests
  test_constraints.cpp   # Constraint tests

bin/
  run_types_demo.cpp     # Types demo
  run_dynamics_demo.cpp  # Dynamics demo
  validate_dynamics.cpp  # Pre-WP2 validation

configs/
  phys.yaml              # Physical parameters
  limits.yaml            # Operational limits
  scales.yaml            # Scaling factors
```

---

## Development Journey

### Phase 1: Types and Config (Task 1)

- Strongly typed `State`, `Control`, `Phys`, `Limits`, `Diag`
- Eigen conversions and validation utilities  
- Config YAMLs: `phys.yaml`, `limits.yaml`, `scales.yaml`
- Unit tests and a types demo notebook (`00-types-visualization.ipynb`)

### Phase 2: Dynamics and Constraints (WP1+)

- Implemented q and n computations; control-surface moments  
- Constraint checking and enforcement; diagnostic logging  
- Expanded tests in `tests/test_constraints.cpp`

### Phase 3: Validation and Smoothness

- Pre-WP2 validation executable `validate_dynamics`  
- Smooth approximations (`smooth_atan2`, clamp/max/min) to prevent NaN  
- Scaling pipeline finished; round-trip checks

---

## Testing & Validation Framework

### Unit Tests (C++)

- `tests/test_types.cpp`: constructors, conversions, validation, error handling
- `tests/test_dynamics.cpp`: dynamics correctness/shape checks
- `tests/test_constraints.cpp`: q/n analytic comparisons, limit triggers, diagnostics
- `tests/simple_test.cpp`: standalone sanity checks

### Demos and Validation

- `bin/run_types_demo.cpp`, `bin/run_dynamics_demo.cpp`: interactive demos
- `bin/validate_dynamics.cpp`: reference flight case + continuity checks  
  - RK4 vs RK45 agreement  
  - Quaternion norm stability  
  - Aerodynamic continuity and wind smoothness

### Visualization

- `notebooks/00-types-visualization.ipynb`: trajectory and diagnostics plots

---

## Current Status & Results

- ✅ All unit tests pass for types/constraints (as documented)  
- ✅ Dynamics/constraints validated against analytic checks  
- ✅ Scaling verified; smooth functions integrated  
- 🔄 Reference plots can be generated from validation CSVs  
- 🔄 Optional logging framework can be added later

Typical nominal values (small sounding rockets):
- `q_max` = 40 kPa; `n_max` = 5 g  
- `CL_alpha` ≈ 3.5 1/rad; `Cm_alpha` ≈ -0.8 1/rad

---

## Lessons Learned

1) Strong typing + Eigen conversions greatly reduce integration errors  
2) Smooth math prevents AD/numerical issues down the line  
3) Early diagnostics (q/n flags) speed up debugging and validation  
4) Scaling to O(1) improves stability and portability across modules

---

## Future Recommendations

- Add logging (e.g., spdlog) and verbose runtime flags  
- Extend aerodynamic models (Mach effects, tabulated coefficients)  
- Add wind and turbulence profiles; parameterize via YAML  
- Build shared library `librocket_physics.so` with pybind11 bindings  
- C++ ↔ Python parity tests on full trajectories

---

## References

- WP1+ Summary: `docs/wp1_plus_summary.md`  
- Task 1 summary: `docs/task1_summary.md`  
- Design: `docs/DESIGN.md`  
- Architecture diagram: `docs/architecture_diagram.md`  
- Configs: `configs/phys.yaml`, `configs/limits.yaml`, `configs/scales.yaml`

---

## WP1 Operations: How to Run

### Build

```bash
cd build
cmake ..
make -j$(nproc) validate_dynamics run_types_demo run_dynamics_demo
```

### Demos

```bash
# Types demo
./bin/run_types_demo

# Dynamics demo (nominal scenario)
./bin/run_dynamics_demo
```

### Pre-WP2 Validation

```bash
# Generate aerodynamic continuity and reference CSVs
./bin/validate_dynamics --all

# Or nominal reference case only
./bin/validate_dynamics --reference
```

Outputs (CSV) can be plotted to verify q(t), n(t), altitude, velocity continuity and limits.

---

## Appendix A: Task 1 — State, Control, and Parameter Representations

Summary of `docs/task1_summary.md` (consolidated):

- Core types in `src/physics/types.hpp` with Eigen conversions
- State (14), Control (T, uT_b, δ), Phys, Limits, Diag
- Config YAMLs: `configs/phys.yaml`, `configs/scales.yaml`, `configs/limits.yaml`
- Tests: `tests/test_types.cpp`, `tests/simple_test.cpp`; demo: `bin/run_types_demo.cpp`
- Notebook: `notebooks/00-types-visualization.ipynb`

Key utilities: `stateDim()=14`, `controlDim()=4/5`, quaternion normalize, validation helpers.

Validation: All unit tests pass; conversions and integrity checks verified.

---

## Appendix B: WP1+ — Structural and Aerodynamic Add-Ins

Summary of `docs/wp1_plus_summary.md` (consolidated):

- Load factor computation: `n = L/(m g0) + 1`, with `L = q S_ref CL_alpha α`
- Enhanced q-limit diagnostics; added `n_violation` flag
- Tail/wing effects via aero coefficients (`CL_alpha`, `Cm_alpha`, `C_delta`, `l_ref`)
- Control surface `delta` in `Control`
- Constraint handling functions: `enforceLimits`, `checkQLimit`, `checkNLimit`
- Unit tests in `tests/test_constraints.cpp`

Config updates: `configs/phys.yaml` (aero params), `configs/limits.yaml` (`q_max`, `n_max`).

Integration: Constraints map directly to WP2 path constraints; penalties can be used for PINN losses.


