# Architecture Changelog

This document tracks all architectural changes to the PINN models, starting from the original baseline and evolving through Directions A, B, C, C1, and C2.

---

## [2025-11-15] Original Baseline PINN Model

### Files
- **`src/models/pinn.py`**
  - Class `PINN`: Vanilla Physics-Informed Neural Network for 6-DOF rocket trajectories
- **`src/models/architectures.py`**
  - `TimeEmbedding` (FourierFeatures): Expands time into 17D features (1 raw + 8 frequency pairs)
  - `ContextEncoder`: Shallow single-layer MLP (7 → 16, Tanh)
  - `MLP`: Feedforward network with configurable depth and width
- **`src/train/losses.py`**
  - `PINNLoss`: Combines data MSE, physics residual (via autograd), and boundary condition loss
- **`configs/pinn_config.yaml`**
  - Default configuration for baseline model
- **`experiments/exp1_17_11_pinn_baseline/`**
  - Training experiment achieving RMSE ≈ 0.9 on normalized states

### Architecture Details

**Baseline PINN - Vanilla MLP Architecture:**
- Input: `(t, context)` where `t` is time grid `[batch, N, 1]` and `context` is 7D parameter vector
- Time embedding: Fourier features (8 frequencies → 17D)
- Context encoding: Shallow encoder (7 → 16, Tanh)
- MLP network: 6 hidden layers × 128 neurons, tanh activation, layer norm, dropout 0.05
- Output: 14D state trajectory `[batch, N, 14]`

**Data Flow:**
```
t → TimeEmbedding → t_emb [17D]
context → ContextEncoder → ctx_emb [16D]
[t_emb || ctx_emb] → MLP(6×128) → state [14D]
```

**Loss Function:**
- `L = λ_data·L_data + λ_phys·L_phys + λ_bc·L_bc` (default: 1.0, 0.1, 1.0)
- Physics loss computed via autograd on predicted trajectory

### Configuration

Default configuration (`configs/pinn_config.yaml`):
```yaml
model:
  type: pinn
  n_hidden: 6
  n_neurons: 128
  activation: tanh
  fourier_features: 8
  context_embedding_dim: 16
loss:
  lambda_data: 1.0
  lambda_phys: 0.1
  lambda_bc: 1.0
```

### Notes
- Serves as the reference baseline for all subsequent architecture directions (A, B, C, C1, C2)
- All later models maintain the same input/output interface: `(t, context) → state [batch, N, 14]`
- Observed limitations: RMSE ≈ 0.9, instability in quaternions, poor temporal structure exploitation, limited context representation (shallow encoder), insufficient physics integration
- These limitations motivated the development of more sophisticated architectures (latent ODE, sequence models, hybrid approaches)

---

## [2025-11-18] Direction A – Latent ODE Implementation

### Added Files
- **`src/models/latent_ode.py`**
  - New class: `LatentDynamicsNet` - Neural network that computes dz/dt in latent space
  - New class: `LatentODEBlock` - Fixed-step Euler integrator for latent ODE
  - New class: `RocketLatentODEPINN` - Main model implementing Direction A architecture

### Modified Files
- **`src/models/__init__.py`**
  - Added exports for `RocketLatentODEPINN`, `LatentDynamicsNet`, `LatentODEBlock`

- **`src/train/train_pinn.py`**
  - Added model type selection logic to support `model_type: latent_ode`
  - Model instantiation now checks `model_cfg.get("type", "pinn")` and creates appropriate model
  - Supports both `pinn` (baseline MLP) and `latent_ode` (Direction A) architectures

### Architecture Details

**Direction A - Latent Neural ODE PINN:**
- Encodes context parameters into initial latent state `z0`
- Evolves latent state through neural ODE: `dz/dt = g_θ(z, t, ctx_emb)`
- Decodes latent trajectory back to physical state `s(t)`
- Uses fixed-step Euler integration (can be upgraded to RK4 later)

**Key Components:**
1. **Context Encoder**: Maps context → context embedding (64 dim)
2. **z0 Encoder**: Maps context embedding → initial latent state (64 dim)
3. **Latent Dynamics Net**: MLP that computes dz/dt (3 layers × 128 neurons)
4. **ODE Integrator**: Fixed-step Euler solver
5. **Decoder**: MLP that maps latent z → physical state s (3 layers × 128 neurons)

### Configuration

To use Direction A, set in config YAML:
```yaml
model:
  type: latent_ode
  latent_dim: 64
  context_embedding_dim: 64
  fourier_features: 8
  dynamics_n_hidden: 3
  dynamics_n_neurons: 128
  decoder_n_hidden: 3
  decoder_n_neurons: 128
  activation: tanh
  layer_norm: true
  dropout: 0.05
```

### Implementation Status
✅ **Completed:**
- All core components implemented and tested
- Model works with both batched and unbatched inputs
- Integration with training pipeline complete
- Configuration files created

### Next Steps
- [ ] Benchmark RMSE on test set and compare with MLP baseline
- [ ] Add debug plots for latent state norm over time
- [ ] Compare trajectories: baseline MLP vs. latent ODE for same context
- [ ] Consider upgrading to RK4 integration if Euler shows stability issues
- [ ] Add unit tests for latent ODE components
- [ ] Run full training experiment with `configs/model_latent_ode.yaml`

### Notes
- All changes are backward compatible - existing `type: pinn` configs continue to work
- Loss functions (`PINNLoss`) require no changes as model still outputs `[batch, N, 14]`
- Data loaders require no changes

---

## [2025-11-18] Direction B – Sequence Transformer PINN

### Added Files
- **`src/models/sequence_pinn.py`**
  - New class `RocketSequencePINN` implementing the Transformer-based sequence architecture.
- **`configs/model_B_sequence.yaml`**
  - Defines model hyperparameters for the sequence Transformer.
- **`configs/train_sequence.yaml`**
  - Provides a smoke-test training configuration for Direction B.

### Modified Files
- **`src/models/__init__.py`**
  - Exported `RocketSequencePINN`.
- **`src/train/train_pinn.py`**
  - Added `model_type: sequence` branch to instantiate the new architecture.

### Notes
- Sequence model treats the entire time grid jointly to capture temporal correlations.
- Uses Fourier time embeddings, context encoder, Transformer encoder, and MLP head.
- Training config uses small batch/epoch counts for quick regression tests; upscale for full runs.

---

## [2025-11-18] Direction C – Hybrid Sequence + Latent ODE PINN

### Added Files
- **`src/models/hybrid_pinn.py`**
  - New class `RocketHybridPINN` combining a Transformer encoder for `z0` with latent ODE dynamics.
- **`configs/model_C_hybrid.yaml`**
  - Captures default hyperparameters for Direction C.
- **`configs/train_hybrid.yaml`**
  - Smoke-test training configuration for the hybrid model.

### Modified Files
- **`src/models/__init__.py`**
  - Exported `RocketHybridPINN`.
- **`src/train/train_pinn.py`**
  - Added `model_type: hybrid` instantiation branch.

### Notes
- Transformer encoder uses the first `encoder_window` timesteps to estimate the latent initial state.
- Latent ODE solver reuses Direction A components for dynamics integration.
- Decoder mirrors Direction A to ensure consistent `[batch, N, 14]` outputs and compatibility with existing losses.

---

## [2025-11-19] Direction C1 – Hybrid Stability + Context Upgrades

### Added / Modified Files
- **`src/models/architectures.py`**
  - Added `DeepContextEncoder`, `OutputHeads`, and `normalize_quaternion`.
- **`src/models/hybrid_pinn.py`**
  - Added `RocketHybridPINNC1` implementing split heads, quaternion normalization, Δ-state decoding, and debugging hooks.
- **`src/models/__init__.py`**
  - Exported the new architecture helpers and `RocketHybridPINNC1`.
- **`src/train/train_pinn.py`**
  - Added helper to pass true `s0` into models that require Δ-state decoding.
  - Registered `model.type: hybrid_c1`.
- **`src/eval/visualize_pinn.py`**, **`run_evaluation.py`**
  - Forward helpers updated to pass initial states and aggregate new diagnostics/metrics.
- **Configs**
  - Added `configs/model_C1_hybrid.yaml` and `configs/train_hybrid_c1.yaml`.

### Architecture Enhancement Sets

Direction C1 introduces two key enhancement sets that address fundamental limitations of the baseline and earlier directions:

#### Set#1: Output Stability Enhancements

**Components:**
- **Split Output Heads**: Separate specialized heads for translation, rotation, and mass
  - Translation head: Predicts `[x, y, z, vx, vy, vz]` (6D)
  - Rotation head: Predicts `[q0, q1, q2, q3, wx, wy, wz]` (7D)
  - Mass head: Predicts `[Δm]` (1D)
- **Quaternion Normalization**: Explicit unit-norm enforcement for quaternions via `normalize_quaternion` function
- **Δ-State Reconstruction**: Predict state deltas and add to initial state (`s = s0 + Δs`)
  - Requires `initial_state` (s0) as input
  - Ensures boundary conditions are satisfied exactly
- **Debug Statistics**: Tracking of quaternion norms, mass monotonicity, delta magnitudes

**Rationale**: Improves training stability, ensures boundary conditions, and prevents invalid quaternion predictions. The Δ-state approach provides a stronger baseline and better conditioning.

#### Set#2: Deep Context Encoder

**Components:**
- **DeepContextEncoder**: Multi-layer MLP replacing shallow single-layer context encoder
  - Architecture: `7 → 64 → 128 → 128 → 64 → 32`
  - Activation: GELU (all hidden layers)
  - Layer normalization: Enabled
  - Output dimension: 32 (configurable)
- Replaces baseline's shallow `ContextEncoder` (7 → 16, single layer)

**Rationale**: Captures complex interactions between context parameters (m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag) through deeper non-linear transformations. Enables better physics regime distinction and parameter interaction modeling.

**Usage**: Integrated into Shared Stem in Direction C2.

### Highlights
- Output head split for translation / rotation / mass plus quaternion normalization.
- Decoder now predicts Δ-state and re-adds `s0` for stability.
- Deep context encoder (64→128→128→64→32 with GELU + LayerNorm) shared across Transformer + dynamics.
- Evaluation logs:
  - Quaternion norms (pre/post normalization)
  - Mass monotonicity violations
  - Δ-state magnitude stats
  - Translation vs rotation RMSE aggregates
- Backward compatible with Direction A/B/C configs; select `model.type: hybrid_c1` for the enhanced variant.

---

## [2025-11-19] Direction C2 – Shared Stem + Dedicated Branches

### Added Files
- **`src/models/shared_stem.py`**
  - New class `SharedStem`: Unified temporal + context processing
  - Combines FourierFeatures (time), DeepContextEncoder (Set#2), and TransformerEncoder (temporal)
  - Outputs shared embedding `[batch, N, hidden_dim]` for all branches

- **`src/models/branches.py`**
  - New class `TranslationBranch`: Specialized MLP for position + velocity (6D)
  - New class `RotationBranch`: Specialized MLP for quaternion + angular velocity (7D)
  - New class `MassBranch`: Specialized MLP for mass delta (1D)
  - Each branch is independent with configurable architecture

- **`src/models/hybrid_pinn.py`**
  - New class `RocketHybridPINNC2`: Complete C2 architecture
  - Integrates Shared Stem → Latent ODE → Dedicated Branches → Δ-state reconstruction

- **`configs/model_C2.yaml`**
  - Model architecture configuration for C2

- **`configs/train_C2.yaml`**
  - Complete training configuration for C2

### Modified Files
- **`src/models/__init__.py`**
  - Added exports for `RocketHybridPINNC2`, `SharedStem`, `TranslationBranch`, `RotationBranch`, `MassBranch`

- **`src/train/train_pinn.py`**
  - Added `model_type: hybrid_c2` instantiation branch
  - Supports all C2 configuration parameters

### Architecture Details

**Direction C2 - Shared Stem + Dedicated Branches:**

The C2 architecture elevates Direction C by introducing:

1. **Shared Stem**: 
   - Time encoding (FourierFeatures, 8 frequencies → 17D)
   - Context encoding (DeepContextEncoder from Set#2, context_dim → 32D)
   - Temporal modeling (TransformerEncoder, 4 layers, 128D hidden)
   - Output: Shared embedding `[batch, N, 128]`

2. **Latent ODE Integration**:
   - Shared embedding used to derive z0 (via Transformer encoder window)
   - Latent ODE evolves z(t) from z0 (reuses Direction C components)
   - Output: Latent trajectory `[batch, N, 64]`

3. **Dedicated Branches**:
   - **TranslationBranch**: `[64] → [128, 128] → [6]` (x, y, z, vx, vy, vz)
   - **RotationBranch**: `[64] → [256, 256] → [7]` (q0, q1, q2, q3, wx, wy, wz)
   - **MassBranch**: `[64] → [64] → [1]` (Δm)

4. **Output Processing** (Set#1):
   - Quaternion normalization via `normalize_quaternion`
   - Δ-state reconstruction: `s = s0 + Δs`
   - Final output: `[batch, N, 14]`

### Key Features

- **Shared Learning**: All branches receive the same shared embedding, enabling global physics learning
- **Specialized Subsystems**: Each branch is optimized for its specific domain (translation/rotation/mass)
- **No Interference**: Branches are completely independent, preventing cross-contamination
- **Full Integration**: Maintains compatibility with Set#1 (quaternion norm, Δ-state) and Set#2 (DeepContextEncoder)
- **Physics Consistency**: Latent ODE ensures physics-informed dynamics

### Configuration

To use Direction C2, set in config YAML:
```yaml
model:
  type: hybrid_c2
  latent_dim: 64
  shared_stem_hidden_dim: 128
  temporal_type: transformer
  temporal_n_layers: 4
  temporal_n_heads: 4
  translation_branch_dims: [128, 128]
  rotation_branch_dims: [256, 256]
  mass_branch_dims: [64]
  dynamics_n_hidden: 3
  dynamics_n_neurons: 128
```

### Implementation Status
✅ **Completed:**
- All core components implemented
- Shared Stem with Transformer temporal modeling
- Dedicated branches for translation, rotation, mass
- Full integration with training pipeline
- Configuration files created
- Comprehensive documentation

### Expected Improvements

Compared to C1 baseline:
- **RMSE Reduction**: 20-30% expected improvement
- **Quaternion Stability**: Norm ≈ 1.0 (within 1e-6)
- **Mass Monotonicity**: No increases (violation < 0.1%)
- **Physics Residual**: Lower ||ds/dt - f(s)||
- **Training Stability**: Smoother loss curves, faster convergence

### Key Design Decisions

1. **Shared Stem Output Dimension**
   - **Decision**: Shared embedding dimension = 128 (configurable)
   - **Rationale**: Balances capacity with computational efficiency
   - **Alternative**: Could use latent_dim from ODE, but keeping separate allows flexibility

2. **Branch Architecture**
   - **Decision**: Separate MLPs with different widths (translation: 128, rotation: 256, mass: 64)
   - **Rationale**: Rotation is most complex (quaternion + angular velocity), mass is simplest
   - **Alternative**: Uniform width, but specialized widths should improve learning

3. **Latent ODE Integration**
   - **Decision**: Use shared embedding to derive z0, then evolve via ODE
   - **Rationale**: Maintains physics consistency from Direction C
   - **Alternative**: Could skip ODE and use shared embedding directly, but loses physics structure

4. **Δ-State Reconstruction**
   - **Decision**: Require s0 and compute s = s0 + Δs
   - **Rationale**: Improves stability (Set#1), ensures boundary conditions
   - **Alternative**: Direct state prediction, but less stable

### Verification Checklist

- [x] Shared Stem processes time + context correctly
- [x] Dedicated branches output correct dimensions
- [x] Quaternion normalization applied
- [x] Δ-state reconstruction works
- [x] Model integrates with training pipeline
- [x] Config files created
- [x] Documentation updated
- [x] Evaluation script supports hybrid_c2
- [ ] Full training run (100 epochs)
- [ ] RMSE comparison with C1
- [ ] Debug plots generated
- [ ] Hyperparameter tuning completed

### Next Steps After Implementation

1. **Hyperparameter Tuning**: Optimize branch widths, shared stem depth
2. **Ablation Studies**: Compare with/without shared stem, with/without branches
3. **Attention Visualization**: If using Transformer, visualize attention maps
4. **Latent State Analysis**: Plot z(t) trajectories, check for exploding/vanishing
5. **Physics Residual Analysis**: Histogram of ||ds/dt - f(s)|| across test set
6. **Branch Output Analysis**: Visualize translation, rotation, mass outputs separately
7. **Shared Embedding Statistics**: Analyze what patterns the shared stem learns

### Notes
- Model requires `initial_state` (s0) for Δ-state reconstruction
- All changes are backward compatible - existing model types continue to work
- Loss functions (`PINNLoss`) require no changes as model still outputs `[batch, N, 14]`
- Data loaders require no changes
- Base architecture: Direction C (Hybrid Model) + Set#1 (Output Stability) + Set#2 (Deep Context Encoder)

---

## [2025-11-24] Direction D – Dependency-Aware Backbone

### Added Files
- **`src/models/direction_d_pinn.py`**
  - Class `DirectionDPINN`: shared backbone + ordered heads (G3 mass → G2 attitude → G1 translation)
  - `TemporalIntegrator` helper (re-used by D1)
- **`configs/model_direction_d.yaml`**
  - Model hyperparameters for Direction D
- **`configs/train_direction_d.yaml`**
  - Training recipe for `direction_d_baseline`

### Modified Files
- **`src/models/__init__.py`** – exports `DirectionDPINN`
- **`src/train/train_pinn.py`** – adds `model_type: direction_d`
- **`run_evaluation.py`** – evaluation support for `direction_d`

### Architecture Details

1. **Feature Encoding**
   - Time: FourierFeatures (8 frequencies → 17D)
   - Context: `ContextEncoder` (7 → 32, GELU + LayerNorm)

2. **Shared Backbone**
   - MLP `[256, 256, 256, 256]` with GELU + LayerNorm
   - Produces latent tensor `[batch, N, 256]`

3. **Dependency Heads**
   - **G3 (Mass):** `[latent] → [128, 64] → 1`
   - **G2 (Attitude + ω):** input `[latent || m]`, MLP `[256, 128, 64] → 7`, quaternion normalized before use
   - **G1 (Translation):** input `[latent || m || q || w]`, MLP `[256, 128, 128, 64] → 6` (position + velocity)
   - Final state: `[x, y, z, vx, vy, vz, q, w, m]`

### Key Features
- Ordered prediction chain enforces physics dependencies (mass informs attitude, both inform translation).
- No latent ODE or Δ-state reconstruction; model does not require `initial_state`.
- Pure MLP stack → significantly faster training/inference (~40% speedup vs. C2).
- Quaternion normalization handled immediately after G2 to keep rotation stable.

### Configuration

```yaml
model:
  type: direction_d
  fourier_features: 8
  context_embedding_dim: 32
  backbone_hidden_dims: [256, 256, 256, 256]
  head_g3_hidden_dims: [128, 64]
  head_g2_hidden_dims: [256, 128, 64]
  head_g1_hidden_dims: [256, 128, 128, 64]
  activation: gelu
  layer_norm: true
  dropout: 0.0
```

### Implementation Status
- ✅ Model + configs + training hooks committed
- ✅ Evaluation script supports `direction_d`
- ✅ Baseline experiment `exp6_24_11_direction_d_baseline` (RMSE ≈ 0.30)
- 🔄 Further tuning and regularization under investigation

### Notes
- Mass monotonicity emerges empirically but is not yet enforced analytically.
- Serves as low-latency alternative to hybrid C-series.

---

## [2025-11-24] Direction D1 – Physics-Aware Dependency Backbone

### Added Files
- **`configs/model_direction_d1.yaml`**
  - Extends Direction D with physics-aware toggles
- **`configs/train_direction_d1.yaml`**
  - Training recipe for `direction_d1_baseline`

### Modified Files
- **`src/models/direction_d_pinn.py`**
  - Adds class `DirectionDPINN_D1`
- **`src/models/__init__.py`**, **`src/train/train_pinn.py`**, **`run_evaluation.py`**
  - Support `model_type: direction_d1`

### Architecture Details

1. **Physics Layer**
   - `PhysicsComputationLayer` computes density, dynamic pressure, and aero coefficients from altitude, velocity, and context.
   - Physics features are concatenated into the G2/G1 inputs.

2. **6D Rotation Representation**
   - G2 predicts 6D rotation + angular velocity (9 outputs) to avoid quaternion normalization gradients.
   - Converted to rotation matrices, then back to quaternions for loss/eval.

3. **Acceleration Head + Temporal Integrator**
   - G1 outputs acceleration instead of Δ-state.
   - `TemporalIntegrator` (RK4 by default) reconstructs velocity/position, optionally seeded with `initial_state`.

### Key Features
- Physics-aware signals guide the ordered heads, reducing the burden on pure data fitting.
- Causal RK4 integration enforces that translation is the time integral of predicted acceleration.
- Still benefits from dependency chain (m → R,ω → accel → integrate).

### Configuration

```yaml
model:
  type: direction_d1
  backbone_hidden_dims: [256, 256, 256, 256]
  head_g3_hidden_dims: [128, 64]
  head_g2_hidden_dims: [256, 128, 64]
  head_g1_hidden_dims: [256, 128, 128, 64]
  integration_method: rk4
  use_physics_aware: true
```

### Implementation Status
- ✅ Model, configs, and training hooks committed
- ✅ `exp7_24_11_direction_d1_baseline` run (RMSE ≈ 0.285)
- 🔄 Need additional regularization for 6D rotation head and physics-layer stability

### Notes
- Still optional to pass `initial_state`; when provided, integrator seeds with `v0`, `z0`.
- Next steps: iterate physics feedback with integrated altitude/velocity; explore soft constraints on mass slope.

---

