# CHECK 3 FAILURE REPORT: True-Attitude Injection Test

**Date:** Generated automatically  
**Experiment:** exp9_25_11_direction_d151_pos_vel_consistency  
**Checkpoint:** best.pt  
**Status:** ❌ **FAILED**

---

## Summary

Check 3 (True-Attitude Injection Test) has **FAILED**. Injecting the true attitude does NOT improve the trajectory, indicating the problem is in the **translational subsystem (G1 head)**, not the rotation subsystem.

---

## Test Configuration

- **Procedure:** Replace predicted quaternion with true quaternion
- **Keep:** Predicted position, velocity, mass, angular velocity
- **Expected:** If translation becomes straight → rotation is the problem
- **Actual:** Translation wobble persists → translation/aero is the problem

---

## Results

### Improvement Analysis
- **Improvement ratio:** 1.00x (NO improvement)
- **Max horizontal deviation (predicted attitude):** 305.4 m
- **Max horizontal deviation (true attitude):** 305.4 m
- **Conclusion:** Attitude is NOT the cause of wobble

### G1 Head Analysis (Normal Cases)

**True Trajectories:**
- Horizontal position: **ZERO** (x, y = 0)
- Horizontal velocity: **ZERO** (vx, vy = 0)

**Predicted Trajectories:**
- Horizontal position: **NON-ZERO**
  - |x| mean: 0.0168 (nondim) → **169 m** when scaled
  - |x| max: 0.0304 (nondim) → **305 m** when scaled
- Horizontal velocity: **NON-ZERO**
  - |vx| mean: 0.2494 (nondim) → **78 m/s** when scaled
  - |vx| max: 0.4370 (nondim) → **137 m/s** when scaled

**Systematic Bias:**
- x bias: **82.5 m** (mean error)
- vx bias: **55.3 m/s** (mean error)

---

## Root Cause Analysis

The failure indicates that the **G1 (translation) head is directly producing horizontal position and velocity** even when the true trajectories have zero horizontal motion.

### Key Findings:

1. **True data has zero horizontal motion** - All test trajectories are perfectly vertical
2. **Model predicts non-zero horizontal motion** - G1 head is producing horizontal bias
3. **Attitude is not the cause** - Injecting true attitude doesn't help
4. **Systematic bias exists** - Mean x error of 82.5 m, mean vx error of 55.3 m/s

### Possible Causes:

1. **Training Data Issue:**
   - Model may not have been trained with sufficient zero-horizontal-motion examples
   - Training data might have bias toward horizontal motion
   - Loss function may not penalize horizontal errors strongly enough

2. **Model Architecture Issue:**
   - G1 head initialization may have bias
   - Context encoding may introduce horizontal bias
   - Backbone features may encode directional drift

3. **Loss Function Issue:**
   - Horizontal position/velocity errors may not be weighted correctly
   - Component weights may favor vertical over horizontal

---

## Diagnostic Files Created

1. **Plot:** `check3_true_attitude_injection_diagnostic.png`
   - Comparison of predicted vs injected trajectories
   - Horizontal deviation over time
   - Improvement ratio analysis

2. **Data:** `check3_true_attitude_injection_diagnostic.json`
   - All numerical results in JSON format

3. **Scripts:**
   - `scripts/diagnostic_check3_true_attitude_injection.py`
   - `scripts/debug_g1_normal_cases.py`

---

## Code Locations to Investigate

### Primary Suspects:

1. **`src/models/direction_d_pinn.py`** - DirectionDPINN_D15 class
   - **Lines 524-560:** G1 head (translation prediction) - **CHECK THIS FIRST**
   - **Lines 499-501:** Context embedding and feature construction
   - **Lines 501:** Backbone feature extraction

2. **`src/train/losses.py`** - PINNLoss class
   - **Lines 104-139:** Data loss computation
   - **Lines 75-94:** Component weights configuration
   - Check if horizontal components (x, y, vx, vy) are weighted correctly

3. **Training Configuration:**
   - Check if training data has horizontal motion
   - Verify loss component weights for horizontal vs vertical

---

## Fix Suggestions

### Immediate Actions:

1. **Inspect G1 Head Architecture:**
   - Check if there's initialization bias
   - Verify input features don't encode horizontal bias
   - Check if context encoding introduces bias

2. **Review Training Data:**
   - Verify if training trajectories have horizontal motion
   - Check if test set is representative
   - Ensure sufficient zero-horizontal-motion examples

3. **Adjust Loss Function:**
   - Increase weight for horizontal position/velocity errors
   - Add explicit penalty for horizontal drift
   - Verify component weights are balanced

4. **Model Architecture:**
   - Consider adding explicit constraint for zero horizontal motion when appropriate
   - Add regularization to penalize horizontal bias
   - Check if backbone is encoding directional drift

### Code Changes to Consider:

```python
# Option 1: Add horizontal drift penalty to loss
lambda_horizontal_drift = 0.1
horizontal_drift_penalty = torch.mean(x_pred[:, :, :2]**2 + v_pred[:, :, :2]**2)
total_loss += lambda_horizontal_drift * horizontal_drift_penalty

# Option 2: Increase component weights for horizontal
component_weights = {
    "x": 2.0,  # Increase from 1.0
    "y": 2.0,  # Increase from 1.0
    "vx": 2.0,  # Increase from 1.0
    "vy": 2.0,  # Increase from 1.0
}
```

---

## Next Steps

1. **Investigate training data** - Check if trajectories should have horizontal motion
2. **Review loss function** - Ensure horizontal errors are properly penalized
3. **Check G1 head initialization** - Verify no systematic bias
4. **Proceed to Check 4** - Horizontal Acceleration Log to understand force computation

---

## Expected Fix Outcome

After fixing, the model should:
- Predict zero horizontal position/velocity when true data has zero
- Match true horizontal motion when it exists
- Show improvement when true attitude is injected (if attitude was also an issue)

---

**Report Generated By:** Diagnostic Script (Check 3)  
**Script Location:** `scripts/diagnostic_check3_true_attitude_injection.py`


