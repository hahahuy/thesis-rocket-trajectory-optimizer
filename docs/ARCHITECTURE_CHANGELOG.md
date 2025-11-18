# Architecture Changelog

This document tracks all architectural changes to the PINN models, following the guidelines in `PINN_v2_Refactor_Directions.md`.

---

## [2025-01-XX] Direction A – Latent ODE Implementation

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

