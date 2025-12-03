# Rocket Trajectory Optimizer - Parameter Documentation

This document provides a comprehensive list of all parameters used in the physics system, data generator, and neural network models.

---

## 1. Physics System Parameters

### 1.1 Aerodynamic Parameters

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Drag coefficient | `Cd` | dimensionless | 0.3 | Drag coefficient |
| Lift coefficient | `Cl` | dimensionless | 0.0 | Lift coefficient |
| Reference area | `S_ref` | m² | 0.05 | Reference area for aerodynamic forces |
| Reference length | `l_ref` | m | 1.2 | Reference length for moment calculations |
| Lift curve slope | `CL_alpha` | 1/rad | 3.5 | Lift coefficient per unit angle of attack |
| Pitch moment coefficient | `Cm_alpha` | 1/rad | -0.8 | Pitch moment coefficient per unit angle of attack |
| Control surface authority | `C_delta` | 1/rad | 0.05 | Control surface effectiveness |
| Maximum control surface deflection | `delta_limit` | rad | 0.1745 (~10°) | Maximum control surface deflection angle |

**Source:** `src/physics/types.hpp`, `configs/phys.yaml`

### 1.2 Inertia Parameters

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Inertia tensor | `I_b` | kg⋅m² | Diagonal [1000, 1000, 100] | 3×3 inertia tensor in body frame |
| Moment of inertia (X-axis) | `Ix` | kg⋅m² | 1000.0 | Principal moment of inertia about X-axis |
| Moment of inertia (Y-axis) | `Iy` | kg⋅m² | 1000.0 | Principal moment of inertia about Y-axis |
| Moment of inertia (Z-axis) | `Iz` | kg⋅m² | 100.0 | Principal moment of inertia about Z-axis |
| Center of gravity offset | `r_cg` | m | [0.0, 0.0, 0.0] | CG offset from body origin |

**Source:** `src/physics/types.hpp`, `configs/phys.yaml`

### 1.3 Propulsion Parameters

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Specific impulse | `Isp` | s | 300.0 | Specific impulse (efficiency measure) |
| Standard gravity | `g0` | m/s² | 9.81 | Standard gravitational acceleration |

**Source:** `src/physics/types.hpp`, `configs/phys.yaml`

### 1.4 Atmospheric Parameters

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Sea level density | `rho0` | kg/m³ | 1.225 | Atmospheric density at sea level |
| Atmospheric scale height | `h_scale` or `H` | m | 8400.0 / 8500.0 | Exponential atmosphere scale height |

**Source:** `src/physics/types.hpp`, `configs/phys.yaml`

### 1.5 Operational Limits

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Maximum thrust | `T_max` | N | 1,000,000 | Maximum allowable thrust |
| Dry mass | `m_dry` | kg | 1000.0 | Minimum mass (after fuel depletion) |
| Maximum dynamic pressure | `q_max` | Pa | 40,000 | Maximum allowable dynamic pressure |
| Maximum angle of attack | `alpha_max` | rad | 0.1 | Maximum allowable angle of attack |
| Maximum gimbal rate | `w_gimbal_max` | rad/s | 1.0 | Maximum gimbal angular velocity |
| Maximum load factor | `n_max` | g | 5.0 | Maximum normal load factor |
| Minimum altitude | `h_min` | m | 0.0 | Minimum allowable altitude |
| Maximum velocity | `v_max` | m/s | 1000.0 | Maximum allowable velocity |

**Source:** `configs/limits.yaml`

### 1.6 Control Parameters

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Thrust rate | `thrust_rate` | N/s | 1e6 | Maximum thrust change rate |
| Gimbal rate | `gimbal_rate_rad` | rad/s | 1.0 | Maximum gimbal angular velocity |

**Source:** `src/data/generator.py`

### 1.7 Wind Parameters

| Parameter | Symbol | Unit | Default Value | Description |
|-----------|--------|------|---------------|-------------|
| Wind magnitude | `wind_mag` | m/s | 0.0 | Wind speed magnitude |
| Wind direction | `wind_dir_rad` | rad | 0.0 | Wind direction angle (azimuth) |
| Wind X-component | `wind_u` | m/s | 0.0 | Wind velocity in X direction (East) |
| Wind Y-component | `wind_v` | m/s | 0.0 | Wind velocity in Y direction (North) |
| Wind Z-component | `wind_w` | m/s | 0.0 | Wind velocity in Z direction (Up) |
| Gust amplitude | `gust_amp` | m/s | 0.0 | Gust amplitude |
| Gust frequency | `gust_freq` | Hz | 1.0 | Gust frequency |
| Gust axis | `gust_axis` | - | "x" | Gust axis direction |
| Gust phase | `gust_phase` | rad | 0.0 | Gust phase offset |
| Wind type | `wind_type` | - | "constant" | Wind model type: "zero", "constant", or "gust" |

**Source:** `src/data/generator.py`, `src/physics/atmosphere.hpp`

---

## 2. Data Generator Parameters

### 2.1 Sampled Parameters (with ranges)

These parameters are randomly sampled during data generation using Latin Hypercube Sampling (LHS) or Sobol sequences.

| Parameter | Symbol | Range | Unit | Description |
|-----------|--------|-------|------|-------------|
| Initial mass | `m0` | [45.0, 65.0] | kg | Initial rocket mass (wet mass) |
| Specific impulse | `Isp` | [220.0, 280.0] | s | Propellant efficiency |
| Drag coefficient | `Cd` | [0.25, 0.45] | - | Drag coefficient |
| Lift curve slope | `CL_alpha` | [2.5, 4.5] | 1/rad | Lift coefficient per unit angle of attack |
| Pitch moment coefficient | `Cm_alpha` | [-1.2, -0.4] | 1/rad | Pitch moment coefficient per unit angle of attack |
| Maximum thrust | `Tmax` | [3000.0, 5000.0] | N | Maximum available thrust |
| Wind magnitude | `wind_mag` | [0.0, 15.0] | m/s | Wind speed magnitude |

**Source:** `configs/dataset.yaml`

### 2.2 Fixed/Default Parameters

These parameters are held constant or use default values during data generation.

| Parameter | Symbol | Default Value | Unit | Description |
|-----------|--------|---------------|------|-------------|
| Reference area | `S` or `S_ref` | 0.05 | m² | Reference area for aerodynamic forces |
| Reference length | `l_ref` | 1.2 | m | Reference length for moments |
| Moment of inertia (X) | `Ix` | 10.0 | kg⋅m² | Principal moment of inertia about X-axis |
| Moment of inertia (Y) | `Iy` | 10.0 | kg⋅m² | Principal moment of inertia about Y-axis |
| Moment of inertia (Z) | `Iz` | 1.0 | kg⋅m² | Principal moment of inertia about Z-axis |
| Sea level density | `rho0` | 1.225 | kg/m³ | Atmospheric density at sea level |
| Atmospheric scale height | `H` | 8500.0 | m | Exponential atmosphere scale height |
| Dry mass | `mdry` | 35.0 | kg | Minimum mass after fuel depletion |
| Maximum gimbal angle | `gimbal_max_rad` | 0.1745 | rad | Maximum gimbal deflection (~10°) |
| Thrust rate | `thrust_rate` | 1e6 | N/s | Maximum thrust change rate |
| Gimbal rate | `gimbal_rate_rad` | 1.0 | rad/s | Maximum gimbal angular velocity |
| Wind direction | `wind_dir_rad` | 0.0 | rad | Wind direction angle (default: from North) |
| Wind type | `wind_type` | "constant" | - | Wind model type |

**Source:** `src/data/generator.py`

### 2.3 Constraints

| Parameter | Symbol | Value | Unit | Description |
|-----------|--------|-------|------|-------------|
| Maximum dynamic pressure | `qmax` | 40,000 | Pa | Maximum allowable dynamic pressure |
| Maximum load factor | `nmax` | 5.0 | g | Maximum normal load factor |

**Source:** `configs/dataset.yaml`

### 2.4 Data Generation Settings

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Number of training cases | `n_train` | 120 | Training dataset size |
| Number of validation cases | `n_val` | 20 | Validation dataset size |
| Number of test cases | `n_test` | 20 | Test dataset size |
| Sampling method | `sampler` | "lhs" | Sampling method: "lhs", "sobol", or "grid" |
| Time horizon | `time_horizon_s` | 30.0 | s | Trajectory duration |
| Grid frequency | `grid_hz` | 50 | Hz | Time grid resolution |

**Source:** `configs/dataset.yaml`

---

## 3. Model Parameters (Context Vector)

The neural network models use a **22-dimensional context vector** that encodes the scenario parameters. This vector is normalized before being fed to the model.

### 3.1 Context Vector Fields (in order)

The context vector is defined in `src/data/preprocess.py` with the following field order:

| Index | Parameter | Symbol | Unit | Description |
|-------|----------|--------|------|-------------|
| 0 | Initial mass | `m0` | kg | Initial rocket mass |
| 1 | Specific impulse | `Isp` | s | Propellant efficiency |
| 2 | Drag coefficient | `Cd` | - | Drag coefficient |
| 3 | Lift curve slope | `CL_alpha` | 1/rad | Lift coefficient per unit angle of attack |
| 4 | Pitch moment coefficient | `Cm_alpha` | 1/rad | Pitch moment coefficient per unit angle of attack |
| 5 | Reference area | `S` | m² | Reference area for aerodynamic forces |
| 6 | Reference length | `l_ref` | m | Reference length for moments |
| 7 | Maximum thrust | `Tmax` | N | Maximum available thrust |
| 8 | Dry mass | `mdry` | kg | Minimum mass after fuel depletion |
| 9 | Maximum gimbal angle | `gimbal_max_rad` | rad | Maximum gimbal deflection |
| 10 | Thrust rate | `thrust_rate` | N/s | Maximum thrust change rate |
| 11 | Gimbal rate | `gimbal_rate_rad` | rad/s | Maximum gimbal angular velocity |
| 12 | Moment of inertia (X) | `Ix` | kg⋅m² | Principal moment of inertia about X-axis |
| 13 | Moment of inertia (Y) | `Iy` | kg⋅m² | Principal moment of inertia about Y-axis |
| 14 | Moment of inertia (Z) | `Iz` | kg⋅m² | Principal moment of inertia about Z-axis |
| 15 | Sea level density | `rho0` | kg/m³ | Atmospheric density at sea level |
| 16 | Atmospheric scale height | `H` | m | Exponential atmosphere scale height |
| 17 | Wind magnitude | `wind_mag` | m/s | Wind speed magnitude |
| 18 | Wind direction | `wind_dir_rad` | rad | Wind direction angle |
| 19 | Gust amplitude | `gust_amp` | m/s | Gust amplitude |
| 20 | Gust frequency | `gust_freq` | Hz | Gust frequency |
| 21 | Maximum dynamic pressure | `qmax` | Pa | Maximum allowable dynamic pressure |
| 22 | Maximum load factor | `nmax` | g | Maximum normal load factor |

**Source:** `src/data/preprocess.py` - `CONTEXT_FIELDS`

### 3.2 Model Input/Output

**Input:**
- **Time:** `t` [nondimensional] - Time grid for trajectory prediction
- **Context:** 22-dimensional normalized vector (from `CONTEXT_FIELDS`)

**Output:**
- **State:** 14-dimensional vector [nondimensional]
  - `[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]`
  - Position (3), Velocity (3), Quaternion (4), Angular velocity (3), Mass (1)

**Source:** `src/models/pinn.py`, `src/models/direction_d_pinn.py`

### 3.3 Normalization

Context parameters are normalized using physics-aware scaling:

- **Mass parameters** (`m0`, `mdry`): Normalized by mass scale `M`
- **Force parameters** (`Tmax`): Normalized by force scale `F`
- **Inertia parameters** (`Ix`, `Iy`, `Iz`): Normalized by `M × l_ref²`
- **Area parameters** (`S`): Normalized by `l_ref²`
- **Velocity parameters** (`wind_mag`, `gust_amp`): Normalized by velocity scale `V`
- **Frequency parameters** (`gust_freq`): Normalized by time scale `T`
- **Dimensionless parameters** (`Cd`, `CL_alpha`, `Cm_alpha`, `nmax`): Already O(1), kept as-is
- **Angle parameters** (`wind_dir_rad`, `gimbal_max_rad`, `gimbal_rate_rad`): Kept in radians
- **Special parameters:**
  - `Isp`: Normalized by reference value (250.0 s)
  - `rho0`: Normalized by sea level density (1.225 kg/m³)
  - `H`: Normalized by reference scale height (8500.0 m)

**Source:** `src/data/preprocess.py` - `build_context_vector()`

---

## 4. Parameter Flow Summary

```
Physics System (25+ parameters)
    ↓
Data Generator (7 sampled + 15 fixed)
    ↓
Context Vector (22 normalized parameters)
    ↓
Neural Network Model
    ↓
State Prediction (14 dimensions)
```

### Key Differences:

1. **Physics System** includes all parameters needed for 6-DOF dynamics simulation
2. **Data Generator** samples 7 key parameters and uses fixed defaults for others
3. **Model Context** uses 22 parameters that define the scenario (subset of physics parameters)

### Notes:

- Not all physics parameters are included in the model context
- Some physics parameters (e.g., `g0`, `C_delta`, `delta_limit`) are fixed or derived internally
- The model learns to predict trajectories based on the 22 context parameters
- Wind parameters are included in context but have minimal effect since OCP solver uses `wind=0`

---

## 5. File References

- **Physics Parameters:** `src/physics/types.hpp`, `configs/phys.yaml`, `configs/limits.yaml`
- **Data Generator:** `src/data/generator.py`, `configs/dataset.yaml`
- **Model Context:** `src/data/preprocess.py` (CONTEXT_FIELDS)
- **Model Architecture:** `src/models/pinn.py`, `src/models/direction_d_pinn.py`

---

*Last updated: 2025-01-02*

