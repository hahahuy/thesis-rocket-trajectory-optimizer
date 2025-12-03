# CHECK 1 FAILURE REPORT: Zero-Aero Zero-Wind Test

**Date:** Generated automatically  
**Experiment:** exp9_25_11_direction_d151_pos_vel_consistency  
**Checkpoint:** best.pt  
**Status:** ❌ **FAILED**

---

## Summary

Check 1 (Zero-Aero Zero-Wind Test) has **FAILED**. The trajectory shows significant horizontal deviation (281.8 m) when it should be a perfect straight vertical line under zero-aero, zero-wind conditions with identity quaternion.

---

## Test Configuration

- **Cd:** Set to 0.0
- **CL_alpha:** Set to 0.0  
- **Cm_alpha:** Set to 0.0
- **wind_mag:** Set to 0.0
- **Quaternion:** Should be identity [1, 0, 0, 0]

**Expected Result:** Perfect straight vertical line (horizontal deviation < 1 m)

---

## Results

### Horizontal Deviation
- **Max horizontal deviation:** 281.847 m (0.282 km)
- **Mean horizontal deviation:** 168.775 m (0.169 km)
- **Final horizontal deviation:** 119.925 m (0.120 km)
- **Threshold:** 1.0 m
- **Status:** ❌ **EXCEEDS THRESHOLD BY 280x**

### Quaternion Analysis
- **Initial quaternion:** [0.9989, -0.0003, -0.0464, 0.0003]
  - Deviation from identity: q1=-0.0464, q2=-0.0464 (significant!)
- **Final quaternion:** [0.9408, 0.0008, -0.3390, 0.0007]
  - Deviation from identity: q0=0.9408 (should be 1.0), q2=-0.3390 (large deviation!)
- **Expected:** [1.0000, 0.0000, 0.0000, 0.0000]

**Observation:** Quaternion is **NOT maintaining identity** and is drifting significantly, especially in the q2 component.

### Vertical Trajectory
- **Initial altitude:** -0.241 km
- **Final altitude:** 32.091 km
- **Range:** 32.332 km
- **Status:** ✓ Vertical motion is present, but with horizontal drift

---

## Root Cause Analysis

The failure indicates that the error is **NOT** coming from physics (aerodynamics/wind), but from the **model's internal structure**. Specifically, one or more of the following components is causing the wobble:

### 1. **Rotation Representation** ⚠️ **LIKELY CULPRIT**
   - Quaternion is deviating from identity even with zero aero/wind
   - q2 component shows large drift (-0.3390 at final time)
   - This suggests the G2 (attitude) head is producing non-zero rotation even when it shouldn't

### 2. **Rotation → World Transform**
   - Even if quaternion is wrong, the transform from body to world coordinates might be incorrect
   - Need to verify rotation matrix computation

### 3. **G1 Translational Head**
   - The translation head (G1) might be producing horizontal position/velocity bias
   - Could be encoding directional drift in the latent features

### 4. **Context Scaling**
   - Context normalization might be incorrect, causing systematic bias
   - Zero values might not be properly handled

### 5. **Coordinate System Handling**
   - Possible sign error or axis mix-up in coordinate transforms
   - World vs body frame confusion

---

## Diagnostic Files Created

1. **Plot:** `check1_zero_aero_diagnostic.png`
   - 3D trajectory visualization
   - Horizontal deviation over time
   - X-Y projection (ground track)
   - Quaternion components vs time

2. **Data:** `check1_zero_aero_diagnostic.json`
   - All numerical results in JSON format

3. **Script:** `scripts/diagnostic_check1_zero_aero.py`
   - Reusable diagnostic script for future checks

---

## Code Locations to Investigate

### Primary Suspects:

1. **`src/models/direction_d_pinn.py`** - DirectionDPINN_D15 class
   - **Line 511-522:** G2 head (attitude prediction) - **CHECK THIS FIRST**
   - **Line 514-518:** 6D rotation to quaternion conversion
   - **Line 524-527:** G1 head (translation prediction)
   - **Line 486-533:** Forward pass logic

2. **`src/models/architectures.py`**
   - Quaternion normalization functions
   - Rotation matrix conversion utilities

3. **`src/data/preprocess.py`**
   - Context vector building and normalization
   - Line 71-120: `build_context_vector()` function

---

## Fix Suggestions

### Immediate Actions:

1. **Inspect G2 Head Output:**
   - Add logging to see what the G2 head is outputting for zero-aero case
   - Check if 6D rotation representation is producing non-zero values
   - Verify that zero context should produce identity rotation

2. **Check Quaternion Normalization:**
   - Verify `normalize_quaternion()` is working correctly
   - Check if there's a bias in the quaternion prediction

3. **Inspect G1 Head:**
   - Check if G1 is producing horizontal position/velocity even with identity quaternion
   - Verify the dependency chain: G3 → G2 → G1

4. **Context Zero-Handling:**
   - Ensure zero values in context are properly normalized
   - Check if the model was trained with zero-aero cases

5. **Add Debug Logging:**
   - Log intermediate outputs from G2 and G1 heads
   - Log rotation matrices computed from quaternions
   - Log world-frame transforms

### Code Changes to Consider:

```python
# In DirectionDPINN_D15.forward(), add debug logging:
if self.training or debug_mode:
    print(f"G2 output (6D): {att_output[..., :6]}")
    print(f"Quaternion: {q_pred}")
    print(f"G1 output (translation): {trans_output}")
```

---

## Next Steps

1. **STOP HERE** - Do not proceed to Check 2 until Check 1 is fixed
2. **Review the diagnostic plot** to visualize the drift pattern
3. **Inspect G2 head outputs** for zero-aero case
4. **Check quaternion initialization** - should start at identity
5. **Verify coordinate system** - ensure no sign errors

---

## Expected Fix Outcome

After fixing, the trajectory should:
- Have horizontal deviation < 1 m
- Maintain quaternion near identity [1, 0, 0, 0]
- Show straight vertical line in 3D plot

---

**Report Generated By:** Diagnostic Script (Check 1)  
**Script Location:** `scripts/diagnostic_check1_zero_aero.py`


