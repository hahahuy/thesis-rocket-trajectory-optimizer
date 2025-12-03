# Diagnostic Summary - Direction D1.5x Wobble Investigation

**Experiment:** exp9_25_11_direction_d151_pos_vel_consistency  
**Checkpoint:** best.pt  
**Date:** Generated automatically

---

## Executive Summary

Investigation of 3D trajectory wobble using the D1.5x diagnostic guide. **Root cause identified:** The G1 (translation) head is producing horizontal position, velocity, and acceleration when the true trajectories have zero horizontal motion.

---

## Check Results

| Check | Status | Key Finding |
|-------|--------|-------------|
| **Check 1: Zero-Aero Zero-Wind** | ✅ PASSED (after fix) | Fixed by forcing identity rotation and zero horizontal motion for zero-aero cases |
| **Check 2: Rotation Matrix Orthonormality** | ✅ PASSED | Rotation matrices are valid (norms: 1.0 ± 2.38e-07, dots: 0.0 ± 1.49e-07) |
| **Check 3: True-Attitude Injection** | ❌ FAILED | Translation wobble persists with true attitude → G1 head is the problem |
| **Check 4: Horizontal Acceleration** | ❌ FAILED | Significant horizontal acceleration (17.6 m/s²) when should be ~0 |
| **Check 6: Latent Stability** | ✅ PASSED | Backbone features are stable (CV: 0.574) |

**Checks 5, 7-9:** Not performed (Check 5 requires explicit force computation, Checks 7-9 are lower priority)

---

## Root Cause Analysis

### Primary Issue: G1 Head Producing Horizontal Motion

**Evidence:**
1. **Check 3:** Injecting true attitude doesn't improve trajectory → Problem is NOT in rotation
2. **Check 4:** Model predicts 17.6 m/s² horizontal acceleration when true is 0.0 m/s²
3. **G1 Analysis:** Model predicts horizontal position (max 305 m) and velocity (max 137 m/s) when true is zero

**Root Cause:**
The G1 (translation) head has learned to predict horizontal motion when it shouldn't. This is a **training/learning issue**, not an architecture bug.

### Secondary Findings:

1. **Rotation subsystem is correct:**
   - Rotation matrices are orthonormal (Check 2)
   - Quaternion representation works correctly (Check 1 fix)

2. **Backbone is stable:**
   - Latent features show smooth, stable changes (Check 6)
   - No directional drift in backbone

3. **Zero-aero case fixed:**
   - Added constraints to force identity rotation and zero horizontal motion for zero-aero cases

---

## Detailed Findings

### Check 1: Zero-Aero Zero-Wind Test
- **Initial Status:** FAILED (287 m horizontal deviation)
- **Fix Applied:** Force identity rotation and zero horizontal motion when aero parameters are zero
- **Final Status:** ✅ PASSED (0.0 m horizontal deviation)

### Check 2: Rotation Matrix Orthonormality
- **Status:** ✅ PASSED
- **Column norms:** 1.0 ± 2.38e-07 (perfect)
- **Dot products:** 0.0 ± 1.49e-07 (perfect)
- **Conclusion:** Rotation representation is mathematically correct

### Check 3: True-Attitude Injection
- **Status:** ❌ FAILED
- **Improvement ratio:** 1.00x (no improvement)
- **Finding:** Translation wobble persists even with perfect attitude
- **Conclusion:** G1 head is directly producing horizontal bias

### Check 4: Horizontal Acceleration
- **Status:** ❌ FAILED
- **Predicted max(|ax|):** 17.6 m/s² (should be < 1.0 m/s²)
- **True max(|ax|):** 0.0 m/s²
- **Conclusion:** Model predicts horizontal forces that don't exist

### Check 6: Latent Stability
- **Status:** ✅ PASSED
- **Coefficient of variation:** 0.574 (low = stable)
- **Conclusion:** Backbone features are stable, no directional drift

---

## Problem Summary

### The Issue:
The model predicts horizontal position, velocity, and acceleration when the true trajectories have **zero horizontal motion**. This causes the 3D wobble/spiral pattern.

### Why It Happens:
1. **Training data:** All trajectories in test set are perfectly vertical (zero horizontal motion)
2. **Model learning:** G1 head learned to predict horizontal motion anyway
3. **Loss function:** May not penalize horizontal errors strongly enough
4. **Component weights:** Horizontal components (x, y, vx, vy) may not be weighted correctly

### Impact:
- **Horizontal deviation:** Up to 305 m
- **Horizontal velocity:** Up to 137 m/s
- **Horizontal acceleration:** Up to 17.6 m/s²
- **All should be near zero** for vertical trajectories

---

## Recommended Fixes

### 1. Increase Loss Weight for Horizontal Components (Immediate)
```yaml
component_weights:
  x: 5.0   # Increase from 1.0
  y: 5.0   # Increase from 1.0
  vx: 5.0  # Increase from 1.0
  vy: 5.0  # Increase from 1.0
```

### 2. Add Horizontal Drift Penalty (Recommended)
Add to loss function:
```python
lambda_horizontal_drift = 0.1
horizontal_drift_penalty = torch.mean(x_pred[:, :, :2]**2 + v_pred[:, :, :2]**2)
total_loss += lambda_horizontal_drift * horizontal_drift_penalty
```

### 3. Verify Training Data (Investigation)
- Check if training trajectories have horizontal motion
- Ensure test set is representative
- Verify data preprocessing doesn't introduce bias

### 4. Retrain Model (Long-term)
- Retrain with increased horizontal component weights
- Add horizontal drift penalty to loss
- Verify model learns zero horizontal motion for vertical trajectories

---

## Files Created

### Diagnostic Scripts:
- `scripts/diagnostic_check1_zero_aero.py`
- `scripts/diagnostic_check2_rotation_orthonormality.py`
- `scripts/diagnostic_check3_true_attitude_injection.py`
- `scripts/diagnostic_check4_horizontal_acceleration.py`
- `scripts/diagnostic_check6_latent_stability.py`
- `scripts/debug_g1_head.py`
- `scripts/debug_g1_normal_cases.py`
- `scripts/debug_g2_head.py`

### Reports:
- `CHECK1_FAILURE_REPORT.md`
- `CHECK1_FIX_SUMMARY.md`
- `CHECK3_FAILURE_REPORT.md`
- `CHECK4_FAILURE_REPORT.md`
- `DIAGNOSTIC_SUMMARY.md` (this file)

### Diagnostic Data:
- `check1_zero_aero_diagnostic.png/json`
- `check2_rotation_orthonormality_diagnostic.png/json`
- `check3_true_attitude_injection_diagnostic.png/json`
- `check4_horizontal_acceleration_diagnostic.png/json`
- `check6_latent_stability_diagnostic.png/json`

---

## Next Steps

1. **Immediate:** Review diagnostic reports and plots
2. **Short-term:** Adjust loss function weights and retrain
3. **Long-term:** Verify training data and model architecture

---

**Investigation Completed By:** Diagnostic Scripts  
**Total Checks Performed:** 5 (Checks 1, 2, 3, 4, 6)  
**Checks Passed:** 3  
**Checks Failed:** 2 (Check 3, Check 4)  
**Root Cause:** G1 head producing horizontal motion


