# Training Improvements Summary

## Changes Made

### 1. Expanded Training Epochs
- **Before:** 120 epochs
- **After:** 160 epochs
- **Location:** `configs/train_direction_d152.yaml`
- **Impact:** Allows more training time, especially for Phase 2 where D1.52 losses are active

### 2. Reduced Phase 1 Ratio
- **Before:** `phase1_ratio: 0.6` (Phase 2 starts at epoch 72)
- **After:** `phase1_ratio: 0.55` (Phase 2 starts at epoch 88)
- **Location:** `configs/train_direction_d152.yaml`
- **Impact:** D1.52 losses activate earlier, giving more time for horizontal suppression training

### 3. Phase-Aware Early Stopping
- **Before:** Single early stopping with patience=15
- **After:** 
  - Phase 1: patience=15
  - Phase 2: patience=40 (configurable via `early_stopping_patience_phase2`)
- **Location:** `configs/train_direction_d152.yaml`, `src/train/train_pinn.py`
- **Impact:** Prevents early stopping from interrupting Phase 2 training before D1.52 losses have full effect

### 4. Composite Metric for Best Checkpoint Selection
- **Before:** Best checkpoint selected based on `val_losses["total"]` only
- **After:** Best checkpoint selected based on composite metric that includes:
  - `val_losses["total"]` (base metric)
  - D1.52 losses: `zero_vxy`, `zero_axy`, `hacc`, `xy_zero` (weighted sum)
- **Location:** `src/train/train_pinn.py`
- **Function:** `compute_composite_metric()`
- **Impact:** Best checkpoint selection now accounts for horizontal suppression, not just data/physics loss

### 5. Log All Loss Components
- **Before:** Only hardcoded loss components logged:
  - `data`, `physics`, `boundary`
  - `mass_residual`, `vz_residual`, `vxy_residual`
  - `smooth_z`, `smooth_vz`
- **After:** ALL loss components from `loss_dict` are logged:
  - Includes: `pos_vel`, `smooth_pos`, `zero_vxy`, `zero_axy`, `hacc`, `xy_zero`
  - Dynamically logs any new loss components added in the future
- **Location:** `src/train/train_pinn.py`
- **Functions:** `train_epoch()`, `validate()`
- **Impact:** Training logs now include D1.52 losses, making it possible to track their progress

### 6. Updated Learning Rate Scheduler
- **Before:** `T_max: 120`
- **After:** `T_max: 160`
- **Location:** `configs/train_direction_d152.yaml`
- **Impact:** Learning rate schedule matches new epoch count

---

## Configuration Changes

### `configs/train_direction_d152.yaml`

```yaml
train:
  epochs: 160  # Increased from 120
  early_stopping_patience: 15
  early_stopping_patience_phase2: 40  # New: Higher patience for Phase 2
  scheduler:
    kwargs:
      T_max: 160  # Updated from 120

loss:
  phase_schedule:
    phase1_ratio: 0.55  # Reduced from 0.6
```

---

## Code Changes

### `src/train/train_pinn.py`

1. **`train_epoch()` function:**
   - Changed from hardcoded `loss_components` dict to dynamic accumulation
   - Now logs ALL keys from `loss_dict` (except "total")

2. **`validate()` function:**
   - Changed from hardcoded `loss_components` dict to dynamic accumulation
   - Now logs ALL keys from `loss_dict` (except "total")

3. **Training loop:**
   - Added `compute_composite_metric()` function
   - Best checkpoint selection uses composite metric instead of just `total` loss
   - Early stopping is phase-aware (uses Phase 2 patience when in Phase 2)
   - Training log includes `val_metric` (composite metric)
   - Print statements include D1.52 losses when present

---

## Expected Benefits

1. **Better Horizontal Suppression:**
   - D1.52 losses activate earlier (epoch 88 vs 72)
   - More training time in Phase 2 (72 epochs vs 32 epochs)
   - Best checkpoint selection accounts for horizontal suppression

2. **Better Training Visibility:**
   - All loss components are logged, including D1.52 losses
   - Can track D1.52 loss progress during training
   - Composite metric shows overall model quality including horizontal suppression

3. **More Robust Training:**
   - Phase 2 early stopping won't trigger too early (patience=40)
   - Training can continue longer to allow D1.52 losses to fully activate
   - Model has more time to learn horizontal suppression

---

## Phase Schedule Details

With `phase1_ratio: 0.55` and `epochs: 160`:
- **Phase 1:** epochs 0-87 (D1.52 losses scaled to 0)
- **Phase 2:** epochs 88-159 (D1.52 losses ramp up with cosine schedule)
- **Phase 2 duration:** 72 epochs (vs 48 epochs before)

**Lambda values at key epochs:**
- Epoch 88: scale ≈ 0.0 (just started Phase 2)
- Epoch 100: scale ≈ 0.15 (15% activation)
- Epoch 120: scale ≈ 0.50 (50% activation)
- Epoch 140: scale ≈ 0.85 (85% activation)
- Epoch 159: scale ≈ 1.0 (100% activation)

---

## Next Steps

1. **Run training** with updated configuration
2. **Monitor training logs** to verify:
   - D1.52 losses are being logged
   - Composite metric is being used for best checkpoint
   - Phase 2 early stopping has higher patience
3. **Evaluate results** to confirm:
   - Better horizontal suppression
   - Best checkpoint is from Phase 2 (or has good horizontal suppression)
   - Training completes Phase 2 without early stopping

---

## Files Modified

1. `configs/train_direction_d152.yaml`
2. `src/train/train_pinn.py`

