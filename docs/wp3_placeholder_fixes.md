# WP3 Placeholder Fixes

## Issues Identified from Sanity Plots

1. **Quaternion initialization**: All zeros instead of `[1,0,0,0]`
2. **Trajectory dynamics**: All flat zeros (no motion)
3. **Parameter plots**: Empty or normalized values instead of SI ranges
4. **Control initialization**: Thrust at t=0 was zero

## Fixes Applied

### 1. Realistic Trajectory Generation

Replaced zero-fill placeholders with a simple vertical ascent simulation:

- **Initialization**: Proper state initialization with identity quaternion `[1,0,0,0]`
- **Dynamics**: Simple vertical ascent with:
  - Gravity: `a = (T/m) - g0`
  - Mass flow: `m_dot = -T / (Isp * g0)`
  - Vertical integration: `vz = vz + a*dt`, `z = z + vz*dt`
- **Quaternion normalization**: Maintained at every step
- **Monitors**: Computed from actual state (rho, q_dyn, n_load)

### 2. Quaternion Initialization

**Before:**
```python
state[0, 6:10] = [0.0, 0.0, 0.0, 0.0]  # Wrong!
```

**After:**
```python
state[0, 6:10] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
# Plus normalization at every step
q_norm = np.linalg.norm(q)
if q_norm > 1e-9:
    state[i, 6:10] = q / q_norm
else:
    state[i, 6:10] = [1.0, 0.0, 0.0, 0.0]
```

### 3. Control Initialization

**Fixed:** Control at t=0 now properly initialized:
```python
control[0, 0] = Tmax * 0.8  # Thrust
control[0, 1:4] = [1.0, 0.0, 0.0]  # Unit direction
```

### 4. Parameter Plotting

**Fixed:** Plotting now uses raw SI values from metadata instead of normalized context vectors:
- Reads `meta/params_used` from raw HDF5 files
- Extracts actual sampled values (m0, Isp, Cd, Tmax) in SI units
- Plots show physical ranges (e.g., 45-65 kg for m0)

### 5. Monitor Calculations

**Fixed:** Load factor now uses actual thrust, not Tmax:
```python
# Before: n_load = sqrt((Tmax/m)^2 + g0^2) / g0
# After:
T_actual = control[:, 0:1]  # Actual thrust at each time
a_total = np.sqrt((T_actual / state[:, 13:14])**2 + g0**2)
n_load = a_total / g0
```

## Verification Results

After fixes, data shows:
- ✅ Altitude: 0 → 27,662 m (rising)
- ✅ Vz: 0 → 2,040 m/s (increasing)
- ✅ Mass: 56.55 → 35.00 kg (decreasing)
- ✅ Quaternion norm: exactly 1.0 at all times
- ✅ Control: Thrust 3,110.8 N, unit direction vectors
- ✅ Monitors: q_dyn max 258,873 Pa, n_load max 9.12 g

## Files Modified

1. `src/data/generator.py`: All three placeholder paths now generate realistic trajectories
2. `scripts/plot_quick_checks.py`: Uses raw SI values from metadata for parameter plots

## Next Steps for WP4

The placeholder data is now realistic enough for:
- Testing the preprocessing pipeline
- Validating schema and scaling
- Visual sanity checks

**Note:** When WP2/WP1 are implemented, these placeholders will be automatically replaced with real OCP solutions and truth integrations.

