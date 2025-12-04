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

## [2025-XX-XX] Direction C3 – Enhanced Hybrid PINN with RMSE Reduction Solutions

### Added Files
- **`src/models/hybrid_pinn.py`**
  - Class `RocketHybridPINNC3`: Enhanced C2 with 6 architectural solutions
- **`src/models/branches.py`**
  - `MonotonicMassBranch`: Structural mass monotonicity (Solution 3)
  - `RotationBranchMinimal`: Quaternion minimal representation (Solution 2)
  - `PhysicsAwareTranslationBranch`: Explicit physics computation (Solution 1)
- **`src/models/latent_ode.py`**
  - `LatentODEBlockRK4`: Higher-order ODE integration (Solution 4)
- **`src/models/coordination.py`**
  - `CoordinatedBranches`: Cross-branch coordination (Solution 5)
  - `AerodynamicCouplingModule`: Aerodynamic coupling module
- **`src/models/physics_layers.py`**
  - `PhysicsComputationLayer`: Explicit density and drag computation
- **`src/models/z0_encoder.py`**
  - `EnhancedZ0Derivation`: Hybrid physics+data z0 initialization (Solution 6)
  - `PhysicsInformedZ0Encoder`: Physics-based z0 encoder

### Modified Files
- **`configs/model_hybrid_c3.yaml`** – C3 model configuration
- **`configs/train_hybrid_c3.yaml`** – C3 training recipe
- **`src/train/train_pinn.py`** – adds `model_type: hybrid_c3`
- **`run_evaluation.py`** – evaluation support for `hybrid_c3`

### Architecture Details

**Base**: Direction C2 (Shared Stem + Dedicated Branches)

**Six Architectural Solutions**:

1. **Solution 1: Physics-Informed Vertical Dynamics Branch**
   - Explicit altitude→density→drag computation
   - `PhysicsComputationLayer` computes `rho(z) = rho0 * exp(-z/H)`
   - Drag force: `F_drag = 0.5 * rho * |v|² * Cd * S`
   - Corrects vertical velocity: `vz_corrected = vz - drag_z / m`

2. **Solution 2: Quaternion Minimal Representation**
   - Rotation vector (3D) → quaternion conversion (always unit norm)
   - Eliminates normalization gradient issues
   - `rotation_vector_to_quaternion()`: axis-angle → quaternion

3. **Solution 3: Structural Mass Monotonicity**
   - `mass_delta = -ReLU(mass_delta_raw)` (always ≤ 0)
   - `mass = cumsum(mass_delta) + m0` (always decreasing)
   - Eliminates 4.2% mass violations from C2

4. **Solution 4: Higher-Order ODE Integration (RK4)**
   - Replaces Euler (O(dt)) with RK4 (O(dt⁴))
   - ~1000x smaller integration error per step
   - 4 function evaluations per step vs Euler's 1

5. **Solution 5: Cross-Branch Coordination**
   - `AerodynamicCouplingModule` computes drag corrections
   - Translation branch receives rotation and mass information
   - Explicit aerodynamic coupling between branches

6. **Solution 6: Enhanced z0 Initialization**
   - Hybrid approach: 30% physics-informed + 70% data-driven
   - Full sequence mean + window Transformer
   - Better initialization reduces error propagation

### Key Features
- **Structural Constraints**: Mass monotonicity and quaternion unit norm guaranteed by architecture
- **Explicit Physics**: Density and drag computed directly, not learned from data
- **Higher-Order Integration**: RK4 reduces integration error accumulation
- **Cross-Branch Coordination**: Aerodynamic coupling between translation and rotation
- **Better Initialization**: Hybrid physics+data z0 reduces error propagation

### Configuration

```yaml
model:
  type: hybrid_c3
  # Same as C2 base parameters
  latent_dim: 64
  fourier_features: 8
  shared_stem_hidden_dim: 128
  # ... (other C2 parameters)
  
  # C3-specific parameters
  z0_blend_alpha: 0.3  # Weight for physics-informed z0
  use_rk4: true        # Use RK4 instead of Euler
  use_physics_aware_translation: true
  use_coordinated_branches: true
```

### Rationale

**C2 Baseline Performance (exp3)**:
- Total RMSE: 0.96
- Rotation RMSE: 0.38 (3.5x worse than exp2)
- Mass violations: 4.2% (physically impossible)
- Vertical dynamics errors: z: 0.91-1.10, vz: 2.98-3.45

**Failed Approach (exp4 - Loss Weighting)**:
- Total RMSE: 1.005 (worse than C2!)
- Mass violations: 4.2% (not fixed)
- **Conclusion**: Loss weighting doesn't fix root causes

**C3 Solution**: Architectural improvements that address root causes structurally, not through penalties.

### Expected Performance

| Component | C2 (exp3) | C3 (Expected) | Improvement | Solution(s) |
|-----------|-----------|---------------|-------------|-------------|
| **Total RMSE** | 0.96 | **0.60-0.75** | 25-40% | All solutions |
| **Translation RMSE** | 1.41 | **0.90-1.20** | 15-30% | Solutions 1, 4, 5 |
| **Rotation RMSE** | 0.38 | **0.15-0.25** | 35-60% | Solutions 2, 5 |
| **Mass RMSE** | 0.19 | **0.10-0.12** | 20-30% | Solution 3 |
| **Vertical (z)** | 0.91-1.10 | **0.60-0.80** | 20-30% | Solutions 1, 4 |
| **Vertical (vz)** | 2.98-3.45 | **1.80-2.40** | 30-40% | Solutions 1, 4 |
| **Mass Violations** | 4.2% | **0%** | 100% fix | Solution 3 |
| **Quaternion Norm** | 1.08 | **1.0** | 100% fix | Solution 2 |

### Implementation Status
- ⚠️ **Planned**: C3 architecture designed but not yet implemented
- 📋 **Reference**: See `docs/expANAL_SOLS.md` for detailed implementation guide
- 🔄 **Next Steps**: Implement solutions in phases (high → medium → low priority)

### Notes
- C3 addresses root causes architecturally, not through loss penalties
- All solutions are backward compatible with C2 base architecture
- Implementation can be done incrementally (solutions can be added one at a time)
- Expected to achieve 25-40% RMSE improvement over C2 baseline

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

## [2025-11-25] Direction D1.5 – Soft-Physics Dependency Backbone

### Added Files
- **`configs/model_direction_d15.yaml`**, **`configs/train_direction_d15.yaml`**
  - Baseline configuration + training recipe for D1.5 experiments.

### Modified Files
- **`src/models/direction_d_pinn.py`**
  - Added `DirectionDPINN_D15` class (shared backbone + optional 6D rotation + mass monotonicity).
- **`src/train/losses.py`**, **`src/train/train_pinn.py`**
  - Loss extensions for soft physics residuals (mass/vz/vxy) and curvature smoothing.
  - Two-phase scheduler to keep physics weights zero during Phase 1.
- **`run_evaluation.py`**, **`src/models/__init__.py`**
  - Model registry + evaluation support for `direction_d15`.
- **`docs/architecture_diagram.md`**
  - Added diagram for Direction D1.5.

### Architecture Details
- Keeps Direction D backbone (Fourier time encoder + context encoder + shared MLP + three heads).
- Optional 6D rotation head (default ON) avoids quaternion normalization issues without needing the full D1 integrator.
- Optional structural mass monotonicity: cumulative sum of negative `softplus` deltas anchored at context-provided `m0`.
- Pairs with soft physics residuals (mass ODE + vertical dynamics) and light smoothing penalties applied in the loss, not the forward pass.
- Two-phase training: first ~75% epochs data-only (physics/smoothing weights = 0), final 25% gradually ramp to low weights (λ ≈ 0.05, smoothing 1e-4).

### Rationale
- Direction D delivered low RMSE but noisy z/vz traces; Direction D1 was smoother but biased mass.
- D1.5 keeps pure data predictions (no integrator) while letting the loss gently penalize non-physical trends, reducing noise without collapsing mass dynamics.

### Implementation Status
- ✅ Code integrated and tested
- ✅ `exp8_25_11_direction_d15_soft_physics` completed (RMSE: **0.199** ✅)
- ✅ Best total RMSE achieved so far
- ✅ Perfect quaternion normalization (norm = 1.0)
- ✅ Excellent mass prediction (RMSE: 0.018)

### Results (exp8)
- **Total RMSE**: 0.199 ✅
- **Translation RMSE**: 0.268
- **Rotation RMSE**: 0.132
- **Mass RMSE**: 0.018 (excellent)
- **Quaternion Norm**: 1.0 (perfect)

### Notes
- Zero-aero handling ensures correct behavior for ballistic trajectories
- Phase schedule (75% data-only) prevents physics loss from interfering with learning
- Soft physics residuals improve trajectory smoothness without integration bias

---

## [2025-11-25] Direction D1.5.1 – Position-Velocity Consistency

### Modified Files
- **`configs/train_direction_d151.yaml`**
  - Added position-velocity consistency loss (`lambda_pos_vel: 0.5`)
  - Added position smoothing penalty (`lambda_smooth_pos: 1e-3`)
  - Adjusted phase schedule (60% data-only vs 75% in D1.5)

### Architecture Details
- **Same architecture as D1.5**: No structural changes
- **Loss Extensions**:
  1. **Position-Velocity Consistency**: Enforces `v = dx/dt` relationship
     - `lambda_pos_vel: 0.5`
     - Ensures predicted velocities match position derivatives
  2. **Position Smoothing**: Second derivative penalty on all positions (x, y, z)
     - `lambda_smooth_pos: 1e-3` (10× stronger than D1.5's z-only smoothing)
     - Reduces oscillations in all position components

### Rationale
- D1.5 achieved excellent RMSE but position-velocity relationship could be more consistent
- Position smoothing helps reduce oscillations observed in D1.5
- Earlier physics introduction (60% vs 75%) allows physics to guide learning sooner

### Implementation Status
- ✅ Code integrated
- ✅ `exp9_25_11_direction_d151_pos_vel_consistency` completed (RMSE: 0.200)

### Results (exp9)
- **Total RMSE**: 0.200
- **Translation RMSE**: 0.270
- **Rotation RMSE**: 0.131 (slightly better than D1.5)
- **Mass RMSE**: 0.019

### Notes
- Similar performance to D1.5
- Position-velocity consistency helps but doesn't dramatically improve overall RMSE
- Position smoothing reduces oscillations but may slightly increase RMSE

---

## [2025-11-26, 2025-11-29] Direction D1.5.2 – Horizontal Motion Suppression

### Modified Files
- **`configs/train_direction_d152.yaml`**
  - Added horizontal motion suppression losses
  - Enhanced component weights for positions
  - Extended training schedule (160 epochs, adjusted phase ratio)

### Architecture Details
- **Same architecture as D1.5**: No structural changes
- **Loss Extensions**:
  1. **Horizontal Velocity Suppression**: `lambda_zero_vxy: 1.0`
     - Penalizes non-zero horizontal velocities (vx, vy)
  2. **Horizontal Acceleration Suppression**: `lambda_zero_axy: 1.0`
     - Penalizes non-zero horizontal accelerations (ax, ay)
  3. **Horizontal Acceleration Penalty**: `lambda_hacc: 0.02`
     - Additional regularization for horizontal motion
  4. **Horizontal Position Penalty**: `lambda_xy_zero: 5.0` (strongest)
     - Forces trajectories to stay near vertical axis
  5. **Enhanced Component Weights**:
     - `x: 2.0`, `y: 2.0`, `z: 3.0` - Boosted position weights
  6. **Extended Training**:
     - 160 epochs (vs 120)
     - Phase 1 ratio: 55% (vs 60% in D1.5.1)
     - Phase 2 early stopping patience: 40 (vs 15)

### Rationale
- Vertical ascent trajectories should have minimal horizontal motion
- Suppressing horizontal motion allows model to focus capacity on vertical dynamics
- Strong penalties ensure trajectories stay near vertical axis

### Implementation Status
- ✅ Code integrated
- ✅ `exp10_26_11_direction_d152_horizontal_suppression` completed (RMSE: 0.200)
- ✅ `exp11_29_11_direction_d152_horizontal_suppression` completed (RMSE: **0.198** ✅✅)

### Results

**exp10 (2025-11-26)**:
- **Total RMSE**: 0.200
- **Translation RMSE**: 0.270
- **Rotation RMSE**: 0.131
- **Mass RMSE**: 0.019

**exp11 (2025-11-29)** - **Best Overall**:
- **Total RMSE**: **0.198** ✅✅ (best achieved)
- **Translation RMSE**: 0.266
- **Rotation RMSE**: 0.132
- **Mass RMSE**: 0.015 (best mass RMSE)
- **Quaternion Norm**: 1.0 (perfect)

### Key Improvements (exp11 vs exp10)
- Z position improved by 14% (0.074 → 0.064)
- Overall translation RMSE improved by 1.4%
- Mass RMSE improved by 21% (0.019 → 0.015)
- Extended training and adjusted phase schedule contributed to improvements

### Notes
- Best total RMSE achieved across all architectures (0.198)
- Horizontal suppression improves vertical dynamics but slightly degrades horizontal positions
- Trade-off between horizontal suppression and horizontal position accuracy
- Perfect quaternion normalization maintained across all D1.5 variants

---

## [2025-12-01] Direction D1.5.3 – V2 Dataloader and Loss Function

### Modified Files
- **`configs/train_direction_d153_v2.yaml`**
  - Enabled v2 dataloader (`use_v2_dataloader: true`)
  - Reduced soft physics loss weights for v2
  - Disabled horizontal motion suppression
  - Adjusted phase schedule parameters

### Architecture Details
- **Same architecture as D1.5**: Uses `DirectionDPINN_D15` model (no structural changes)
- **V2 Dataloader Integration**:
  - Uses `RocketDatasetV2` instead of `RocketDataset`
  - Loads additional features: `T_mag` (thrust magnitude) and `q_dyn` (dynamic pressure)
  - Data source: `data/processed_v2/` instead of `data/processed/`
  - Models receive `T_mag` and `q_dyn` in batches but currently ignore them (future: v2 model versions will use `InputBlockV2`)

- **V2 Loss Function Adjustments**:
  1. **Reduced Soft Physics Weights** (vs D1.5.2):
     - `lambda_mass_residual: 0.025` (vs 0.05 in D1.5.2)
     - `lambda_vz_residual: 0.025` (vs 0.05 in D1.5.2)
     - `lambda_vxy_residual: 0.005` (vs 0.01 in D1.5.2)
     - `lambda_smooth_z: 5.0e-5` (vs 1.0e-4 in D1.5.2)
     - `lambda_smooth_vz: 1.0e-5` (vs 1.0e-4 in D1.5.2)
     - **Rationale**: V2 features (T_mag, q_dyn) provide physics information directly, reducing need for strong physics loss penalties

  2. **Position-Velocity Consistency**:
     - `lambda_pos_vel: 0.5` (same as D1.5.1)
     - `lambda_smooth_pos: 0.0` (disabled, vs 2.0e-3 in D1.5.3 non-v2)

  3. **Horizontal Motion Suppression** (Disabled):
     - All horizontal suppression losses set to 0.0:
       - `lambda_zero_vxy: 0.0`
       - `lambda_zero_axy: 0.0`
       - `lambda_hacc: 0.0`
       - `lambda_xy_zero: 0.0`
     - **Rationale**: V2 features should naturally help with horizontal motion, so explicit suppression not needed

  4. **Phase Schedule**:
     - `phase1_ratio: 0.55` (55% data-only, 45% physics ramp)
     - `ramp: cosine` (smooth transition)
     - Extended training: 160 epochs
     - Phase 2 early stopping patience: 40

### Rationale
- **V2 Dataloader**: Provides physics-critical features (T_mag, q_dyn) directly to models, reducing information bottleneck
- **Reduced Loss Weights**: V2 features encode physics information, so explicit physics loss penalties can be lighter
- **Disabled Horizontal Suppression**: V2 features should naturally guide model toward correct horizontal behavior
- **Future-Proof**: Prepares for v2 model versions that will actually use T_mag and q_dyn via `InputBlockV2`

### Key Features
- **V2 Data Pipeline**: Uses processed_v2 data with T_mag and q_dyn features
- **Lighter Physics Losses**: Reduced weights since v2 features provide physics information
- **Simplified Loss Function**: Disabled horizontal suppression and position smoothing
- **Backward Compatible**: Model accepts but ignores v2 features (no errors)

### Configuration

```yaml
train:
  use_v2_dataloader: true  # Enable v2 dataloader
  data_dir: data/processed_v2  # Use v2 processed data

loss:
  # Reduced soft physics weights for v2
  lambda_mass_residual: 0.025
  lambda_vz_residual: 0.025
  lambda_vxy_residual: 0.005
  lambda_smooth_z: 5.0e-5
  lambda_smooth_vz: 1.0e-5
  
  # Position-velocity consistency (enabled)
  lambda_pos_vel: 0.5
  lambda_smooth_pos: 0.0  # Disabled
  
  # Horizontal suppression (all disabled)
  lambda_zero_vxy: 0.0
  lambda_zero_axy: 0.0
  lambda_hacc: 0.0
  lambda_xy_zero: 0.0
```

### Implementation Status
- ✅ Code integrated
- ✅ `configs/train_direction_d153_v2.yaml` created
- ✅ V2 dataloader support in training script
- ⚠️ Models currently ignore T_mag and q_dyn (loaded but unused)
- 🔄 Future: Create v2 model versions with `InputBlockV2` to actually use v2 features

### Notes
- **Current Limitation**: Models accept T_mag and q_dyn but don't use them in forward pass
- **To Actually Use V2 Features**: Need to create v2 model versions (e.g., `DirectionDPINN_D15_V2`) that use `InputBlockV2`
- **Data Requirements**: Requires v2 preprocessing (`python -m src.data.preprocess_v2`)
- **Backward Compatible**: V2 dataloader gracefully handles missing v2 features (returns zeros)
- **Training**: Proceeds normally with v2 dataloader, just with additional (unused) features in batches

---

## [2025-12-02] Direction AN – Shared Stem + Mission Branches + Physics Residuals

### Added Files
- **`src/models/direction_an_pinn.py`**
  - Class `DirectionANPINN`: Shared stem + mission branches + physics residual layer
  - Class `ANSharedStem`: Residual MLP stem with Fourier features + context encoder
  - Class `DirectionANPINN_AN1`: V2 version with `InputBlockV2` (future implementation)
- **`src/physics/physics_residual_layer.py`**
  - Class `PhysicsResidualLayer`: Computes physics residuals using autograd
  - Class `PhysicsResiduals`: Dataclass for physics residual outputs
- **`configs/train_an.yaml`** – Baseline AN training recipe
- **`configs/train_an_v2.yaml`** – AN with v2 dataloader

### Modified Files
- **`src/models/__init__.py`** – exports `DirectionANPINN`, `DirectionANPINN_AN1`
- **`src/train/train_pinn.py`** – adds `model_type: direction_an`
- **`run_evaluation.py`** – evaluation support for `direction_an`
- **`docs/architecture_diagram.md`** – added AN architecture diagram

### Architecture Details

1. **ANSharedStem**:
   - Time: FourierFeatures (8 frequencies → 17D)
   - Context: ContextEncoder (7 → 128, Tanh)
   - Residual MLP stack: 4 layers × 128, Tanh, LayerNorm
   - Output: `[batch, N, 128]` latent features
   - **Key Feature**: Residual connections for gradient flow stability

2. **Mission Branches** (Independent):
   - **TranslationBranch**: `[128→128→128→6]` for `[x, y, z, vx, vy, vz]`
   - **RotationBranch**: `[128→256→256→7]` for `[q0, q1, q2, q3, wx, wy, wz]`
   - **MassBranch**: `[128→64→1]` for `[m]`
   - **Key Difference**: All branches receive same latent (no dependency chain)

3. **Physics Residual Layer**:
   - Uses `compute_dynamics` from physics library (WP1)
   - Computes ODE residuals using autograd
   - Returns `PhysicsResiduals` dataclass
   - **Purpose**: Provides physics residuals directly from forward pass

4. **Dual Output**:
   - Returns `(state_pred, physics_residuals)`
   - Enables flexible loss computation
   - Physics residuals available for analysis

### Key Features
- **Residual Stem**: Residual MLP stack for better gradient flow
- **Independent Branches**: No dependency chain (simpler than D1.5)
- **Explicit Physics**: Physics residuals computed in forward pass
- **Unified Processing**: Single shared stem processes all features
- **Physics Integration**: Uses actual physics library (WP1 consistency)

### Configuration

```yaml
model:
  type: direction_an
  fourier_features: 8
  stem_hidden_dim: 128
  stem_layers: 4
  activation: tanh
  layer_norm: true
  translation_branch_dims: [128, 128]
  rotation_branch_dims: [256, 256]
  mass_branch_dims: [64]
  dropout: 0.0
```

### Rationale
- **Alternative to Dependency Chain**: Independent branches avoid complexity of ordered heads
- **Residual Connections**: Better gradient flow than plain MLP
- **Explicit Physics**: Physics residuals computed directly, not just in loss
- **Unified Stem**: Simpler than C2's Transformer-based Shared Stem
- **Physics Consistency**: Uses WP1 physics library for residual computation

### Implementation Status
- ✅ Code integrated and tested
- ✅ `exp15_02_12_direction_an_baseline` completed (RMSE: **0.197** ✅)
- ✅ `exp16_04_12_direction_an_v2` completed (RMSE: **0.197** ✅)
- ✅ `exp17_04_12_direction_an_v2` completed (RMSE: **0.197** ✅)
- ✅ V2 dataloader support (models accept but ignore T_mag, q_dyn)
- 🔄 Future: `DirectionANPINN_AN1` will use `InputBlockV2` to fuse v2 features

### Results

**exp15 (2025-12-02) - Baseline**:
- **Total RMSE**: **0.197** ✅
- **Translation RMSE**: 0.264
- **Rotation RMSE**: 0.133
- **Mass RMSE**: 0.015 (excellent)
- **Quaternion Norm**: 1.0 (perfect)

**exp16-17 (2025-12-04) - V2**:
- **Total RMSE**: **0.197** ✅ (consistent with baseline)
- Same performance metrics as exp15
- V2 dataloader works correctly (features loaded but unused)

### Notes
- **No Dependency Chain**: Unlike D1.5, branches are independent (may miss physics dependencies)
- **Simpler than C2**: No Transformer or attention mechanism
- **Physics Residual Overhead**: Computing residuals in forward pass adds computational cost
- **Excellent Performance**: Matches D1.5.2/D1.5.3 performance (0.197 RMSE)
- **V2 Support**: Accepts v2 features but doesn't use them yet (future: AN1 will use InputBlockV2)

---

