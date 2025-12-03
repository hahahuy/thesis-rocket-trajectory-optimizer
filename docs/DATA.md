# Data Structure and Processing Guide

This document describes the data structures used in the rocket trajectory optimizer project, including raw data, processed v1, and processed v2 formats.

## Table of Contents

1. [Overview](#overview)
2. [Raw Data Structure](#raw-data-structure)
3. [Processed v1 Structure](#processed-v1-structure)
4. [Processed v2 Structure](#processed-v2-structure)
5. [Key Differences: v1 vs v2](#key-differences-v1-vs-v2)
6. [Preprocessing](#preprocessing)
7. [Data Loading](#data-loading)
8. [Model Integration](#model-integration)
9. [Migration Guide](#migration-guide)

---

## Overview

The project uses a three-stage data pipeline:

1. **Raw Data** (`data/raw/`): Individual case files from WP3 dataset generation
2. **Processed v1** (`data/processed/`): Consolidated splits with time, context, and state
3. **Processed v2** (`data/processed_v2/`): Extended v1 with physics features (T_mag, q_dyn)

**Key Principle**: v2 is **backward compatible** with v1. All v1 functionality is preserved, and v2 adds optional features that models can use if supported.

---

## Raw Data Structure

**Location**: `data/raw/`

**Format**: Individual HDF5 files per case: `case_train_*.h5`, `case_val_*.h5`, `case_test_*.h5`

**Schema**:
```
case_*.h5
├── time: [N]                    # Time grid [s] (dimensional)
├── state: [N, 14]              # State trajectory (dimensional)
│   [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
├── control: [N, 4]             # Control trajectory (dimensional)
│   [T, uTx, uTy, uTz]
├── monitors/                    # Optional monitoring data
│   ├── q_dyn: [N]              # Dynamic pressure
│   ├── n_load: [N]             # Load factor
│   └── ...
├── ocp/                         # Optional OCP solver data
│   └── ...
└── meta/                        # Metadata
    ├── params_used: JSON string # Physical parameters
    ├── seed: int                # Random seed
    └── ...
```

**Implementation**: See `src/data/storage.py` for raw data I/O utilities.

---

## Processed v1 Structure

**Location**: `data/processed/`

**Format**: Consolidated HDF5 files per split: `train.h5`, `val.h5`, `test.h5`

**Schema**:
```
train.h5 (or val.h5, test.h5)
├── inputs/
│   ├── t: [n_cases, N]         # Time grid (nondimensional)
│   └── context: [n_cases, context_dim]  # Context parameters (normalized)
├── targets/
│   └── state: [n_cases, N, 14]  # State trajectories (nondimensional)
└── meta/
    ├── scales: JSON string      # Reference scales (L, V, T, M, F, W)
    └── context_fields: JSON array  # Context field names
```

**Dimensions**:
- `n_cases`: Number of cases in split (typically 120 train, 20 val, 20 test)
- `N`: Number of time points per trajectory (typically 1501 for 30s at 50Hz)
- `context_dim`: Dimension of context vector (varies, typically 7-20 fields)

**State Format** (14D):
```
[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
```

**Context Fields** (canonical order):
```python
CONTEXT_FIELDS = [
    "m0", "Isp", "Cd", "CL_alpha", "Cm_alpha", "S", "l_ref",
    "Tmax", "mdry", "gimbal_max_rad", "thrust_rate", "gimbal_rate_rad",
    "Ix", "Iy", "Iz", "rho0", "H",
    "wind_mag", "wind_dir_rad", "gust_amp", "gust_freq",
    "qmax", "nmax",
]
```

**Context Vector Field Details**:

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

**Note**: Only fields present in the raw data are included in the context vector. Missing fields are set to 0.0.

**Context Vector Normalization**:

Context parameters are normalized using physics-aware scaling before being fed to models:

- **Mass parameters** (`m0`, `mdry`): Normalized by mass scale `M`
- **Force parameters** (`Tmax`): Normalized by force scale `F`
- **Inertia parameters** (`Ix`, `Iy`, `Iz`): Normalized by `M × l_ref²`
- **Area parameters** (`S`): Normalized by `l_ref²`
- **Velocity parameters** (`wind_mag`, `gust_amp`): Normalized by velocity scale `V`
- **Frequency parameters** (`gust_freq`): Normalized by time scale `T`
- **Dimensionless parameters** (`Cd`, `CL_alpha`, `Cm_alpha`, `nmax`): Already O(1), kept as-is
- **Angle parameters** (`wind_dir_rad`, `gimbal_max_rad`, `gimbal_rate_rad`): Kept in radians
- **Special parameters**:
  - `Isp`: Normalized by reference value (250.0 s)
  - `rho0`: Normalized by sea level density (1.225 kg/m³)
  - `H`: Normalized by reference scale height (8500.0 m)

**Source**: `src/data/preprocess.py` - `build_context_vector()`

**Preprocessing**: See `src/data/preprocess.py` - `process_raw_to_splits()`

**Data Loading**: See `src/utils/loaders.py` - `RocketDataset`, `create_dataloaders()`

---

## Processed v2 Structure

**Location**: `data/processed_v2/`

**Format**: Extended v1 format with additional physics features

**Schema**:
```
train.h5 (or val.h5, test.h5)
├── inputs/
│   ├── t: [n_cases, N]         # Time grid (nondimensional) [v1]
│   ├── context: [n_cases, context_dim]  # Context parameters [v1]
│   ├── T_mag: [n_cases, N]     # Thrust magnitude (nondimensional) [v2 NEW]
│   └── q_dyn: [n_cases, N]     # Dynamic pressure (nondimensional) [v2 NEW]
├── targets/
│   └── state: [n_cases, N, 14]  # State trajectories [v1]
└── meta/
    ├── scales: JSON string      # Reference scales [v1]
    ├── context_fields: JSON array  # Context field names [v1]
    └── version: "v2"            # Version marker [v2 NEW]
```

**V2 Features**:

1. **T_mag (Thrust Magnitude)**:
   - **Computation**: Extracted from control vector `u[:, 0]` (thrust component)
   - **Units**: Nondimensionalized using `F` scale: `T_mag_nd = T_mag / scales.F`
   - **Shape**: `[n_cases, N]`
   - **Purpose**: Provides time-varying thrust information directly to models

2. **q_dyn (Dynamic Pressure)**:
   - **Computation**: `q_dyn = 0.5 * ρ(z) * |v|²`
     - Density: `ρ(z) = ρ₀ * exp(-z / h_scale)` (exponential atmosphere)
     - Speed: `|v| = sqrt(vx² + vy² + vz²)`
   - **Units**: Nondimensionalized using pressure scale: `q_dyn_nd = q_dyn / (F / L²)`
   - **Shape**: `[n_cases, N]`
   - **Purpose**: Provides aerodynamic loading information directly to models

**Preprocessing**: See `src/data/preprocess_v2.py` - `process_raw_to_splits_v2()`

**Data Loading**: See `src/utils/loaders_v2.py` - `RocketDatasetV2`, `create_dataloaders_v2()`

---

## Key Differences: v1 vs v2

### Structural Differences

| Feature | v1 | v2 |
|---------|----|----|
| **Time** | ✅ `inputs/t` | ✅ `inputs/t` (same) |
| **Context** | ✅ `inputs/context` | ✅ `inputs/context` (same) |
| **State** | ✅ `targets/state` | ✅ `targets/state` (same) |
| **T_mag** | ❌ Not included | ✅ `inputs/T_mag` (new) |
| **q_dyn** | ❌ Not included | ✅ `inputs/q_dyn` (new) |
| **Version marker** | ❌ None | ✅ `meta/version: "v2"` |

### Dataloader Differences

**v1 Batch Format** (`RocketDataset`):
```python
{
    "t": [batch, N],           # Time
    "context": [batch, context_dim],  # Context
    "state": [batch, N, 14],   # State
    "case_id": [batch]         # Case index
}
```

**v2 Batch Format** (`RocketDatasetV2`):
```python
{
    "t": [batch, N],           # Time (same as v1)
    "context": [batch, context_dim],  # Context (same as v1)
    "state": [batch, N, 14],   # State (same as v1)
    "T_mag": [batch, N],       # Thrust magnitude (v2 NEW)
    "q_dyn": [batch, N],       # Dynamic pressure (v2 NEW)
    "case_id": [batch]         # Case index (same as v1)
}
```

**Backward Compatibility**: `RocketDatasetV2` gracefully handles missing v2 features (returns zeros) for backward compatibility.

### Model Integration Differences

**v1 Models**:
- Accept: `(t, context)`
- Use: `FourierFeatures` + `ContextEncoder` separately
- Example: `DirectionDPINN`, `DirectionDPINN_D1`, `DirectionDPINN_D15`

**v2 Models** (when implemented):
- Accept: `(t, context, T_mag, q_dyn)`
- Use: `InputBlockV2` to fuse all features
- Example: `DirectionDPINN_D15_V2` (to be created)

**Current Status**: v1 models accept but ignore `T_mag` and `q_dyn` (no errors, but features unused).

---

## Preprocessing

### v1 Preprocessing

**Script**: `src/data/preprocess.py`

**Command**:
```bash
python -m src.data.preprocess \
    --raw data/raw \
    --out data/processed \
    --scales configs/scales.yaml
```

**Process**:
1. Discovers case files in `data/raw/`
2. Splits by naming convention (`case_train_*`, `case_val_*`, `case_test_*`)
3. Extracts context fields from metadata
4. Nondimensionalizes state and control using scales
5. Builds normalized context vectors
6. Saves consolidated HDF5 files per split

**Output**: `data/processed/train.h5`, `val.h5`, `test.h5`

### v2 Preprocessing

**Script**: `src/data/preprocess_v2.py`

**Command**:
```bash
python -m src.data.preprocess_v2 \
    --raw data/raw \
    --out data/processed_v2 \
    --scales configs/scales.yaml
```

**Process**:
1. Same as v1 preprocessing (steps 1-6)
2. **Additional**: Computes `T_mag` from control vector
3. **Additional**: Computes `q_dyn` from state (altitude, velocity)
4. **Additional**: Nondimensionalizes `T_mag` and `q_dyn`
5. **Additional**: Saves v2 features to HDF5
6. **Additional**: Adds `meta/version: "v2"` marker

**Output**: `data/processed_v2/train.h5`, `val.h5`, `test.h5`

**Key Functions**:
- `compute_thrust_magnitude(u)`: Extracts `T = u[:, 0]` from control
- `compute_dynamic_pressure(state, scales, rho0, h_scale)`: Computes `q = 0.5 * ρ * |v|²`

**Implementation**: See `src/data/preprocess_v2.py:44-83` for computation details.

---

## Data Loading

### v1 Dataloader

**Module**: `src/utils/loaders.py`

**Classes**:
- `RocketDataset`: Dataset class for v1 HDF5 files
- `create_dataloaders()`: Factory function for train/val/test loaders

**Usage**:
```python
from src.utils.loaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data/processed",
    batch_size=8,
    num_workers=0,
    time_subsample=None
)

# Batch format
batch = next(iter(train_loader))
# batch["t"]: [batch, N]
# batch["context"]: [batch, context_dim]
# batch["state"]: [batch, N, 14]
# batch["case_id"]: [batch]
```

**Configuration** (training config):
```yaml
train:
  use_v2_dataloader: false  # or omit (defaults to false)
  data_dir: data/processed
```

### v2 Dataloader

**Module**: `src/utils/loaders_v2.py`

**Classes**:
- `RocketDatasetV2`: Dataset class for v2 HDF5 files
- `create_dataloaders_v2()`: Factory function for train/val/test loaders

**Usage**:
```python
from src.utils.loaders_v2 import create_dataloaders_v2

train_loader, val_loader, test_loader = create_dataloaders_v2(
    data_dir="data/processed_v2",
    batch_size=8,
    num_workers=0,
    time_subsample=None
)

# Batch format
batch = next(iter(train_loader))
# batch["t"]: [batch, N]
# batch["context"]: [batch, context_dim]
# batch["state"]: [batch, N, 14]
# batch["T_mag"]: [batch, N]  # v2 feature
# batch["q_dyn"]: [batch, N]  # v2 feature
# batch["case_id"]: [batch]
```

**Configuration** (training config):
```yaml
train:
  use_v2_dataloader: true  # Enable v2 dataloader
  data_dir: data/processed_v2
```

**Backward Compatibility**: `RocketDatasetV2` checks for v2 features and falls back to zeros if missing, ensuring compatibility with v1 files.

---

## Model Integration

### v1 Models

**Current Models** (v1 compatible):
- `DirectionDPINN` (Direction D)
- `DirectionDPINN_D1` (Direction D1)
- `DirectionDPINN_D15` (Direction D1.5, D1.5.1, D1.5.2)
- `DirectionANPINN` (Direction AN)
- All C-series models (Baseline, A, B, C, C1, C2, C3)

**Forward Pass**:
```python
def forward(self, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    # Process time and context
    t_emb = self.fourier_encoder(t)
    ctx_emb = self.context_encoder(context)
    features = torch.cat([t_emb, ctx_emb], dim=-1)
    # ... rest of model ...
```

**Training Script Integration**: Models receive only `t` and `context` from v1 batches.

### v2 Models (Future)

**Planned Models**:
- `DirectionDPINN_D15_V2`: v2 version of D1.5 with `InputBlockV2`
- `DirectionANPINN_V2`: v2 version of AN with `InputBlockV2`

**Input Block**: `src/models/input_block_v2.py` - `InputBlockV2`

**Architecture**:
```python
class InputBlockV2(nn.Module):
    def __init__(self, context_dim, fourier_features=8, ...):
        self.time_embed = FourierFeatures(n_frequencies=fourier_features)
        self.context_embed = ContextEncoder(...)
        self.extra_embed = nn.Sequential(
            nn.Linear(2, extra_embedding_dim),  # T_mag + q_dyn -> embedding
            nn.GELU()
        )
        # Optional projection
        self.proj = nn.Linear(fused_dim, output_dim)
    
    def forward(self, t, context, T_mag, q_dyn):
        t_emb = self.time_embed(t)  # [batch, N, 17]
        ctx_emb = self.context_embed(context)  # [batch, N, 32]
        extra_emb = self.extra_embed(torch.stack([T_mag, q_dyn], dim=-1))  # [batch, N, 16]
        fused = torch.cat([t_emb, ctx_emb, extra_emb], dim=-1)  # [batch, N, 65]
        return self.proj(fused)  # [batch, N, output_dim]
```

**Forward Pass**:
```python
def forward(self, t, context, T_mag=None, q_dyn=None):
    # Use InputBlockV2 to fuse all features
    features = self.input_block(t, context, T_mag, q_dyn)
    # ... rest of model ...
```

**Training Script Integration**: Training script automatically passes `T_mag` and `q_dyn` from v2 batches if available.

**Current Status**: v1 models accept but ignore `T_mag` and `q_dyn` (no errors). To actually use v2 features, create v2 model versions with `InputBlockV2`.

---

## Migration Guide

### Using v1 Data

**For existing models** (all current models):
1. Use v1 preprocessing: `python -m src.data.preprocess --raw data/raw --out data/processed`
2. Set `use_v2_dataloader: false` (or omit) in training config
3. Use `data_dir: data/processed`
4. Models work as before

### Using v2 Data

**For future v2 models** (when implemented):
1. Use v2 preprocessing: `python -m src.data.preprocess_v2 --raw data/raw --out data/processed_v2`
2. Set `use_v2_dataloader: true` in training config
3. Use `data_dir: data/processed_v2`
4. Create v2 model versions that use `InputBlockV2`

**For current models with v2 data** (backward compatible):
1. Use v2 preprocessing to generate `data/processed_v2/`
2. Set `use_v2_dataloader: true` in training config
3. Models will receive `T_mag` and `q_dyn` but ignore them
4. Training proceeds normally (v2 features unused)

### Verification

**Check v1 data**:
```python
import h5py
f = h5py.File("data/processed/train.h5", "r")
assert "inputs/t" in f
assert "inputs/context" in f
assert "targets/state" in f
assert "inputs/T_mag" not in f  # v1 doesn't have this
print("✓ v1 format confirmed")
```

**Check v2 data**:
```python
import h5py
f = h5py.File("data/processed_v2/train.h5", "r")
assert "inputs/t" in f
assert "inputs/context" in f
assert "targets/state" in f
assert "inputs/T_mag" in f  # v2 has this
assert "inputs/q_dyn" in f  # v2 has this
assert f["meta/version"][()].decode() == "v2"
print("✓ v2 format confirmed")
```

---

## Code References

### Preprocessing

- **v1**: `src/data/preprocess.py`
  - `process_raw_to_splits()`: Main preprocessing function
  - `to_nd()`, `from_nd()`: Nondimensionalization utilities
  - `build_context_vector()`: Context vector construction

- **v2**: `src/data/preprocess_v2.py`
  - `process_raw_to_splits_v2()`: Extended preprocessing function
  - `compute_thrust_magnitude()`: T_mag computation
  - `compute_dynamic_pressure()`: q_dyn computation

### Data Loading

- **v1**: `src/utils/loaders.py`
  - `RocketDataset`: v1 dataset class
  - `create_dataloaders()`: v1 dataloader factory

- **v2**: `src/utils/loaders_v2.py`
  - `RocketDatasetV2`: v2 dataset class
  - `create_dataloaders_v2()`: v2 dataloader factory

### Model Integration

- **v1 Input Processing**: `src/models/architectures.py`
  - `FourierFeatures`: Time embedding
  - `ContextEncoder`: Context embedding

- **v2 Input Processing**: `src/models/input_block_v2.py`
  - `InputBlockV2`: Unified input fusion block

### Training Integration

- **Training Script**: `src/train/train_pinn.py`
  - Automatically selects v1 or v2 dataloader based on config
  - Passes `T_mag` and `q_dyn` to models if available (models can ignore)

---

## Summary

| Aspect | v1 | v2 |
|--------|----|----|
| **Location** | `data/processed/` | `data/processed_v2/` |
| **Features** | t, context, state | t, context, state, **T_mag, q_dyn** |
| **Dataloader** | `RocketDataset` | `RocketDatasetV2` |
| **Config Flag** | `use_v2_dataloader: false` | `use_v2_dataloader: true` |
| **Models** | All current models | Future v2 models (with `InputBlockV2`) |
| **Backward Compat** | N/A | ✅ v2 dataloader works with v1 files |
| **Status** | ✅ Production | ✅ Ready (models ignore v2 features) |

**Recommendation**: Use v1 for current models. Use v2 when creating new model versions that explicitly use `T_mag` and `q_dyn` via `InputBlockV2`.

---

## References

- **WP3 Comprehensive**: [wp3_comprehensive_description.md](wp3_comprehensive_description.md) - Dataset generation details
- **Scaling Guide**: [configs/scales.yaml](../configs/scales.yaml) - Reference scales for nondimensionalization
- **Phase Schedule**: [PHASE_SCHEDULE_EXPLANATION.md](PHASE_SCHEDULE_EXPLANATION.md) - Training phase schedule details

