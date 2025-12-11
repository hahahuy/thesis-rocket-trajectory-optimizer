# Development Stage of the PINN model

## Table of Contents
1. [Data](#data)
2. [Model Structure](#model-structure)
   - 1. [Base Model](#1-base-model)
   - 2. [Direction A: Latent Neural ODE PINN](#2-direction-a-latent-neural-ode-pinn)
   - 3. [Direction B: Sequence Model (Transformer) PINN](#3-direction-b-sequence-model-transformer-pinn)
   - 4. [Direction C: Hybrid PINN](#4-direction-c-hybrid-pinn)
   - 5. [Direction C1: Enhanced Hybrid PINN](#5-direction-c1-enhanced-hybrid-pinn)
   - 6. [Direction C2: Shared Stem + Dedicated Branches](#6-direction-c2-shared-stem--dedicated-branches)
     - 6.5. [Direction C3: Enhanced Hybrid PINN with RMSE Reduction Solutions](#65-direction-c3-enhanced-hybrid-pinn-with-rmse-reduction-solutions)
   - 7. [Direction D: Dependency-Aware Backbone + Causal Heads](#7-direction-d-dependency-aware-backbone--causal-heads)
     - 7.6. [Direction D1.5: Soft-Physics Dependency Backbone](#76-direction-d15-soft-physics-dependency-backbone)
     - 7.7. [Direction D1.5.1: Position-Velocity Consistency](#77-direction-d151-position-velocity-consistency)
     - 7.8. [Direction D1.5.2: Horizontal Motion Suppression](#78-direction-d152-horizontal-motion-suppression)
     - 7.9. [Direction D1.5.3: V2 Dataloader and Loss Function](#79-direction-d153-v2-dataloader-and-loss-function)
     - 7.10. [Direction D1.5.4: Central Difference Derivative Method](#710-direction-d154-central-difference-derivative-method)
   - 8. [Direction AN: Shared Stem + Mission Branches + Physics Residuals](#8-direction-an-shared-stem--mission-branches--physics-residuals)

---

## Data
start with data **Inputs**
- `t`: the nondimensional time grid for each trajectory (`[batch, N, 1]` or `[N, 1]`). Internally it’s expanded into Fourier features so the network can “see” periodic structure.
- `context`: the scenario-specific parameter vector (here 7 numbers per case: `m0`, `Isp`, `Cd`, `CL_alpha`, `Cm_alpha`, `Tmax`, `wind_mag`). It’s normalized using the stored scales and encoded to a fixed-length embedding.  
  For batched trajectories, context is broadcast to every time step so the model knows which physics regime each sample belongs to.

and data **Outputs**
- `state`: a full 14-dimensional rocket state for every time step in the grid (`[batch, N, 14]` or `[N, 14]`). The component order is fixed:  
  `[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]` (nondimensional).  
  These outputs are what the PINN compares against the ground truth trajectories (data loss), enforces dynamics on (physics loss), and checks initial conditions for (boundary loss).

---

 Note: Different architectures (baseline MLP, latent ODE, transformer sequence, hybrid) just change how they process time/context internally:
- The vanilla PINN concatenates Fourier time features + context embedding and pushes through an MLP.
- The latent-ODE version encodes context into an initial latent state z₀, integrates a neural ODE across the provided time grid, then decodes to the 14-D state.
- The sequence model treats `[t_emb || ctx_emb]` as tokens and runs them through a Transformer before a head that outputs state vectors.

But regardless of variant, the interface is “time grid + context → 14-D state trajectory.”

---

## Model Structure
1. Base model
The base PINN model (`src/models/pinn.py`) is a vanilla Physics-Informed Neural Network that maps time and context parameters directly to the full 14-dimensional rocket state. It uses a simple feedforward architecture with Fourier feature embeddings for time and a shallow context encoder.

### 1.2 Data Intake Pipeline

The model processes data through the following stages:

#### Stage 1: Dataset Loading (`src/utils/loaders.py`)
- **Input Format**: HDF5 files with structure:
  - `inputs/t`: `[n_cases, N]` - nondimensional time grid (typically N=1501 points per trajectory)
  - `inputs/context`: `[n_cases, 7]` - normalized context parameters (`m0`, `Isp`, `Cd`, `CL_alpha`, `Cm_alpha`, `Tmax`, `wind_mag`)
  - `targets/state`: `[n_cases, N, 14]` - ground truth state trajectories (nondimensional)
- **Preprocessing**: Data is pre-loaded into memory, supports optional time subsampling for faster training
- **Output**: Each batch item contains:
  - `t`: `[N]` time grid
  - `context`: `[7]` context vector (normalized)
  - `state`: `[N, 14]` target trajectory

#### Stage 2: Time Embedding (`TimeEmbedding` → `FourierFeatures`)
- **Input**: `t` with shape `[batch, N, 1]` or `[N, 1]` (nondimensional time)
- **Process**: Expands time into Fourier features:
  ```
  t → [t, sin(2π·1·t), cos(2π·1·t), sin(2π·2·t), cos(2π·2·t), ..., sin(2π·8·t), cos(2π·8·t)]
  ```
- **Output**: `[batch, N, 17]` where 17 = 1 (raw time) + 2×8 (8 frequency pairs)
- **Purpose**: Enables the network to capture periodic and multi-scale temporal patterns

#### Stage 3: Context Encoding (`ContextEncoder`)
- **Input**: `context` with shape `[batch, 7]` (normalized context parameters)
- **Broadcasting**: If time is `[batch, N, 1]`, context is expanded to `[batch, N, 7]` by broadcasting
- **Process**: Single linear layer + tanh activation:
  ```
  context [batch, N, 7] → Linear(7 → 16) → Tanh → ctx_emb [batch, N, 16]
  ```
- **Output**: `[batch, N, 16]` context embedding
- **Purpose**: Encodes scenario-specific physics parameters into a fixed-size representation

#### Stage 4: Feature Concatenation
- **Input**: 
  - `t_emb`: `[batch, N, 17]` (time features)
  - `ctx_emb`: `[batch, N, 16]` (context embedding)
- **Process**: Concatenation along feature dimension
- **Output**: `[batch, N, 33]` combined feature vector

#### Stage 5: MLP Network
- **Input**: `[batch, N, 33]` (time + context features)
- **Architecture**: 
  - 6 hidden layers, each with 128 neurons
  - Activation: `tanh` (except output layer: identity)
  - Layer normalization: enabled (on hidden layers)
  - Dropout: 0.05 (on hidden layers)
- **Output**: `[batch, N, 14]` predicted state trajectory
- **Structure**:
  ```
  Input [33] → Hidden1 [128] → Hidden2 [128] → ... → Hidden6 [128] → Output [14]
  ```

### 1.3 Complete Data Flow Diagram

```
Dataset (HDF5)
    │
    ├─ t: [batch, N] ──────────────┐
    ├─ context: [batch, 7] ────────┤
    └─ state: [batch, N, 14] ──────┘ (ground truth, used in loss)
         │
         │
    ┌────▼─────────────────────────┐
    │  DataLoader                   │
    │  (RocketDataset)              │
    └────┬─────────────────────────┘
         │
         ├─ t: [batch, N, 1] ────────────────────┐
         │                                         │
         │                                         ▼
         │                                  ┌──────────────┐
         │                                  │ TimeEmbedding│
         │                                  │ (Fourier)    │
         │                                  └──────┬───────┘
         │                                         │
         │                                         │ t_emb: [batch, N, 17]
         │                                         │
         └─ context: [batch, 7] ──────────────────┤
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ ContextEncoder  │
                                          │ (Linear+Tanh)   │
                                          └────────┬─────────┘
                                                   │
                                                   │ ctx_emb: [batch, N, 16]
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ Concatenate    │
                                          │ [t_emb || ctx] │
                                          └────────┬────────┘
                                                   │
                                                   │ x: [batch, N, 33]
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ MLP Network    │
                                          │ 6×128 (tanh)   │
                                          └────────┬────────┘
                                                   │
                                                   │ state_pred: [batch, N, 14]
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ Loss Function  │
                                          │ (Data+Phys+BC) │
                                          └────────────────┘
```

### 1.4 Loss Function Structure

The model is trained with a composite loss (`src/train/losses.py`):

```
L_total = λ_data · L_data + λ_phys · L_phys + λ_bc · L_bc
```

- **Data Loss** (`L_data`): Mean squared error between predicted and ground truth states
  ```
  L_data = MSE(state_pred, state_true)
  ```

- **Physics Loss** (`L_phys`): ODE residual computed via automatic differentiation
  ```
  r = ∂state_pred/∂t - f(state_pred, control, params)
  L_phys = mean(r²)
  ```
  Where `f` is the 6-DOF rocket dynamics function (`src/physics/dynamics_pytorch.py`)

- **Boundary Loss** (`L_bc`): Initial condition enforcement
  ```
  L_bc = ||state_pred(t=0) - state_true(t=0)||²
  ```

- **Default Weights**: `λ_data = 1.0`, `λ_phys = 0.1`, `λ_bc = 1.0`

### 1.5 Why the Base Model Fails to Give Good Results

The base PINN model achieves RMSE ≈ 0.9 (approximately 90% error on normalized states) and shows instability, particularly in orientation and coupled dynamics. The fundamental limitations are:

#### 1. **Lack of Temporal Structure Exploitation**
- **Problem**: The model treats each time step independently. While Fourier features help with periodic patterns, the MLP processes `[t_emb || ctx_emb]` pointwise without any explicit temporal dependencies.
- **Impact**: The model cannot learn sequential patterns, causal relationships, or long-range dependencies across the trajectory. For a 6-DOF rocket system with complex dynamics, the state at time `t` heavily depends on the history `[0, t)`, but the base model has no mechanism to access or model this history.

#### 2. **Insufficient Physics Integration**
- **Problem**: Physics loss is computed via autograd on the predicted trajectory, but the network architecture itself doesn't encode any inductive bias about the ODE structure. The MLP must learn to satisfy physics constraints purely through gradient-based optimization.
- **Impact**: For complex coupled dynamics (translation-rotation coupling, aerodynamic forces, mass depletion), the model struggles to satisfy physics constraints simultaneously with data fitting. The physics loss acts as a soft constraint rather than a structural guarantee.

#### 3. **Limited Context Representation**
- **Problem**: Context encoding is extremely shallow (single linear layer: 7 → 16). The 7 context parameters (`m0`, `Isp`, `Cd`, `CL_alpha`, `Cm_alpha`, `Tmax`, `wind_mag`) have complex interactions that affect the entire trajectory, but the simple encoder cannot capture these relationships.
- **Impact**: The model cannot effectively distinguish between different physics regimes. For example, high `wind_mag` with low `Cd` vs. low `wind_mag` with high `Cd` may produce similar intermediate states but different overall trajectories—the shallow encoder cannot capture these nuanced differences.

#### 4. **Architectural Limitations**
- **Problem**: 
  - No skip connections or residual pathways
  - Fixed-width MLP (128 neurons) may be insufficient for the complexity
  - Tanh activation can suffer from vanishing gradients for deep networks
  - No specialized output heads for different state components (translation, rotation, mass)
- **Impact**: The model struggles with:
  - **Orientation instability**: Quaternion dynamics (q0, q1, q2, q3) require unit-norm constraints and are highly nonlinear
  - **Coupled dynamics**: Angular velocity (wx, wy, wz) couples with orientation and translation through aerodynamic forces
  - **Mass depletion**: Mass `m` decreases monotonically, but the model has no inductive bias to enforce this

#### 5. **Training Dynamics Issues**
- **Problem**: The loss landscape is highly non-convex with competing objectives (data fitting vs. physics satisfaction). With `λ_phys = 0.1`, physics constraints are relatively weak, leading to trajectories that fit data but violate physics, or vice versa.
- **Impact**: The model gets stuck in poor local minima where it either:
  - Overfits to data but produces unphysical trajectories
  - Satisfies physics approximately but has high data error
  - Fails to balance both objectives, resulting in overall poor performance

#### 6. **No Explicit State Evolution Mechanism**
- **Problem**: The model predicts the entire trajectory in one forward pass without any iterative refinement or evolution mechanism. It must learn the mapping `(t, context) → state(t)` directly.
- **Impact**: For long trajectories (N=1501 points), the model must maintain consistency across all time steps simultaneously. Small errors at early times compound, and there's no mechanism to correct or refine predictions based on intermediate states.

### 1.6 Summary

The base PINN model is a straightforward feedforward architecture that attempts to learn the direct mapping from time and context to state. While conceptually simple and easy to implement, it fails to capture the temporal structure, physics constraints, and complex parameter interactions inherent in 6-DOF rocket trajectories. The observed RMSE ≈ 0.9 and instability issues stem from these architectural limitations, motivating the development of more sophisticated models (latent ODE, sequence models, hybrid architectures) that better exploit temporal structure and physics knowledge.

---

## 2. Direction A: Latent Neural ODE PINN

### 2.1 What Changed Compared to Base Model

**Key Architectural Changes:**
- **Replaced direct MLP mapping** with a three-stage pipeline: encode → evolve → decode
- **Added latent space**: Introduces an intermediate latent representation `z(t)` of dimension 64 (configurable)
- **Neural ODE integration**: Replaces pointwise MLP with explicit ODE integration using Euler method
- **Context-to-z0 encoder**: Context embedding is mapped to initial latent state `z0` via a simple MLP (Linear + Tanh)
- **Latent dynamics network**: New `LatentDynamicsNet` computes `dz/dt = g_θ(z, t_emb, ctx_emb)` in latent space
- **Decoder network**: Maps latent trajectory `z(t)` back to physical state `s(t)`

**Preserved Components:**
- Same time embedding (Fourier features)
- Same context encoder structure (but with larger embedding dimension: 64 vs 16)
- Same loss function structure (data + physics + boundary)
- Same input/output interface: `(t, context) → state`

### 2.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    └─ context: [batch, 7] ──────────────────┤
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ TimeEmbedding    │
                                    │ (Fourier)        │
                                    └────────┬─────────┘
                                             │
                                             │ t_emb: [batch, N, 17]
                                             │
                                    ┌────────▼─────────┐
                                    │ ContextEncoder   │
                                    │ (Linear+Tanh)    │
                                    └────────┬─────────┘
                                             │
                                             │ ctx_emb: [batch, 64]
                                             │
                                    ┌────────▼─────────┐
                                    │ z0_encoder       │
                                    │ (Linear+Tanh)    │
                                    └────────┬─────────┘
                                             │
                                             │ z0: [batch, 64]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ LatentODEBlock                   │
                                    │ (Euler Integration)              │
                                    │                                  │
                                    │ For each time step i:            │
                                    │   condition_i = [t_emb_i || ctx] │
                                    │   dz/dt = dynamics_net(z_i, cond)│
                                    │   z_{i+1} = z_i + dt * dz/dt    │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z_traj: [batch, N, 64]
                                             │
                                    ┌────────▼─────────┐
                                    │ Decoder (MLP)    │
                                    │ 3×128 (tanh)     │
                                    └────────┬─────────┘
                                             │
                                             │ state_pred: [batch, N, 14]
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Loss Function    │
                                    │ (Data+Phys+BC)   │
                                    └──────────────────┘
```

### 2.3 Why It Would Be Better

#### 1. **Explicit ODE Structure**
- **Inductive Bias**: The architecture explicitly models state evolution through `dz/dt = g_θ(z, t, ctx)`, aligning with the physical ODE structure. This provides a strong inductive bias that trajectories should follow continuous dynamics.
- **Physics Consistency**: By integrating through time using an ODE solver, the model naturally enforces temporal consistency. Small errors don't compound as dramatically because the ODE structure constrains the evolution.

#### 2. **Latent Space Compression**
- **Dimensionality Reduction**: The 14-D physical state is compressed to a 64-D latent space, allowing the model to learn a more compact representation of trajectory dynamics.
- **Smoother Optimization**: Learning in latent space can be easier than directly predicting high-dimensional physical states, as the latent representation can capture essential dynamics patterns.

#### 3. **Temporal Causality**
- **Sequential Evolution**: Unlike the base model that predicts all time steps independently, the latent ODE model evolves state sequentially: `z(t_{i+1})` depends explicitly on `z(t_i)` through integration.
- **Long-Range Dependencies**: The ODE integration naturally propagates information across the entire trajectory, allowing early-time information to influence later predictions.

#### 4. **Better Physics Integration**
- **Structural Guarantee**: The ODE structure means that if the latent dynamics network learns to approximate the true dynamics, the integrated trajectory will automatically satisfy physics constraints (up to integration error).
- **Reduced Physics Loss**: With proper training, the physics loss should be lower because the architecture is designed to satisfy ODE constraints structurally.

### 2.4 Why It's Not Working / Giving Good Results

#### 1. **Integration Error Accumulation**
- **Problem**: Fixed-step Euler integration accumulates errors over long trajectories (N=1501 points). Each integration step introduces truncation error, which compounds over time.
- **Impact**: Even if the latent dynamics network is accurate, the numerical integration errors cause the predicted trajectory to drift from the true solution, especially for long trajectories.

#### 2. **Latent Space Mismatch**
- **Problem**: The decoder must learn to map from a 64-D latent space to 14-D physical state. If the latent representation doesn't capture all necessary information, or if the decoder is insufficient, reconstruction errors occur.
- **Impact**: The model may learn a good latent dynamics representation but fail to accurately decode to physical states, leading to high data loss.

#### 3. **z0 Initialization Issues**
- **Problem**: The initial latent state `z0` is derived solely from context embedding via a simple linear layer. This may not capture enough information about the initial physical state, especially for complex initial conditions.
- **Impact**: If `z0` is poorly initialized, the entire trajectory evolution starts from the wrong point, and even perfect latent dynamics cannot recover.

#### 4. **Training Instability**
- **Problem**: Training a neural ODE requires backpropagating through the integration loop, which can lead to:
  - Vanishing/exploding gradients through many time steps
  - Difficulty balancing data loss (on decoded states) with physics loss (on latent dynamics)
- **Impact**: The model may fail to converge or converge to poor local minima where either the latent dynamics or decoder is inaccurate.

#### 5. **Limited Context Encoding**
- **Problem**: While context embedding dimension is increased (64 vs 16), the z0 encoder is still very simple (single linear layer). Complex context parameter interactions are not well captured.
- **Impact**: Different physics regimes may map to similar `z0` values, causing the model to produce similar trajectories for different scenarios.

#### 6. **No Explicit Physics in Latent Dynamics**
- **Problem**: The `LatentDynamicsNet` is a generic MLP that learns `dz/dt` purely from data. It doesn't have any built-in knowledge of the physical dynamics structure.
- **Impact**: The model must learn the entire dynamics from scratch, which is difficult for complex 6-DOF systems with coupled translation-rotation dynamics.

---

## 3. Direction B: Sequence Model (Transformer) PINN

### 3.1 What Changed Compared to Base Model

**Key Architectural Changes:**
- **Transformer encoder**: Replaces the pointwise MLP with a Transformer encoder that processes the entire time sequence
- **Sequence modeling**: Treats `[t_emb || ctx_emb]` as a sequence of tokens, enabling explicit temporal attention
- **Input projection**: Projects concatenated embeddings to Transformer dimension (d_model=128) before sequence processing
- **Output head**: MLP head processes Transformer outputs to produce state predictions
- **Larger context embedding**: Context embedding dimension increased to 64 (vs 16 in base model)

**Preserved Components:**
- Same time embedding (Fourier features)
- Same context encoder structure (but with larger dimension)
- Same loss function structure
- Same input/output interface

**Architecture Details:**
- Transformer encoder: 4 layers, 4 attention heads, feedforward dimension 512
- GELU activation (instead of tanh) for Transformer layers
- Output head: 2-layer MLP (128 → 128 → 14)

### 3.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    └─ context: [batch, 7] ──────────────────┤
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ TimeEmbedding    │
                                    │ (Fourier)        │
                                    └────────┬─────────┘
                                             │
                                             │ t_emb: [batch, N, 17]
                                             │
                                    ┌────────▼─────────┐
                                    │ ContextEncoder  │
                                    │ (Linear+Tanh)   │
                                    └────────┬─────────┘
                                             │
                                             │ ctx_emb: [batch, N, 64]
                                             │
                                    ┌────────▼─────────┐
                                    │ Concatenate     │
                                    │ [t_emb || ctx]  │
                                    └────────┬─────────┘
                                             │
                                             │ seq: [batch, N, 81]
                                             │
                                    ┌────────▼─────────┐
                                    │ Input Projection │
                                    │ Linear(81→128)  │
                                    └────────┬─────────┘
                                             │
                                             │ x: [batch, N, 128]
                                             │
                                    ┌────────▼──────────────────────┐
                                    │ Transformer Encoder          │
                                    │ (4 layers, 4 heads)          │
                                    │                              │
                                    │ Self-attention across time:  │
                                    │ - Each token attends to all  │
                                    │   other tokens in sequence   │
                                    │ - Captures temporal patterns │
                                    └────────┬─────────────────────┘
                                             │
                                             │ x_out: [batch, N, 128]
                                             │
                                    ┌────────▼─────────┐
                                    │ Output Head     │
                                    │ MLP(128→128→14)  │
                                    └────────┬─────────┘
                                             │
                                             │ state_pred: [batch, N, 14]
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Loss Function    │
                                    │ (Data+Phys+BC)   │
                                    └──────────────────┘
```

### 3.3 Why It Would Be Better

#### 1. **Explicit Temporal Dependencies**
- **Self-Attention Mechanism**: The Transformer encoder allows each time step to attend to all other time steps in the sequence, enabling the model to learn long-range temporal dependencies.
- **Causal Awareness**: Unlike the base model that treats time steps independently, the sequence model explicitly models relationships between `s(t_i)` and `s(t_j)` for all `i, j`.

#### 2. **Pattern Recognition**
- **Temporal Patterns**: The attention mechanism can identify and leverage temporal patterns in the trajectory, such as:
  - Acceleration phases vs. coasting phases
  - Rotation dynamics correlated with translation
  - Mass depletion effects over time
- **Multi-Scale Features**: Different attention heads can focus on different temporal scales (short-term vs. long-term dependencies).

#### 3. **Better Context Integration**
- **Context at Every Step**: Context is broadcast to every time step and processed through the Transformer, allowing the model to condition the entire trajectory on physics parameters.
- **Attention to Context**: The self-attention mechanism can learn which time steps are most relevant for different context parameters.

#### 4. **Smoother Trajectories**
- **Global Consistency**: By processing the entire sequence together, the model can enforce global consistency across the trajectory, reducing abrupt jumps or discontinuities.
- **Temporal Smoothness**: The Transformer's ability to attend across the sequence helps produce smoother, more physically plausible trajectories.

### 3.4 Why It's Not Working / Giving Good Results

#### 1. **Quadratic Complexity**
- **Problem**: Self-attention has O(N²) complexity in sequence length. For N=1501 time steps, this requires significant memory and computation.
- **Impact**: 
  - Training is slow and memory-intensive
  - May require gradient checkpointing or sequence truncation
  - Limits batch size, which can hurt training stability

#### 2. **Lack of Physics Inductive Bias**
- **Problem**: The Transformer is a generic sequence model with no built-in knowledge of ODE structure or physics constraints. It must learn everything from data.
- **Impact**: The model may learn spurious temporal correlations that don't correspond to physical dynamics, leading to trajectories that look smooth but violate physics.

#### 3. **Attention Dilution**
- **Problem**: With 1501 time steps, the attention mechanism may struggle to focus on relevant dependencies. Important relationships may be diluted across many time steps.
- **Impact**: The model may fail to capture critical temporal dependencies, especially for long-range effects (e.g., early mass depletion affecting late trajectory).

#### 4. **Physics Loss Challenges**
- **Problem**: Computing physics loss requires autograd through the Transformer, which is computationally expensive and can lead to gradient issues.
- **Impact**: 
  - Physics loss may be difficult to optimize effectively
  - Gradient flow through many attention layers can be unstable
  - May require careful loss weighting and gradient clipping

#### 5. **No Explicit State Evolution**
- **Problem**: Unlike the latent ODE model, the Transformer doesn't explicitly model state evolution. It learns a mapping from time/context to state but doesn't enforce that states evolve according to dynamics.
- **Impact**: The model may produce trajectories that satisfy data loss but violate physics constraints, especially for complex coupled dynamics.

#### 6. **Limited Positional Information**
- **Problem**: While Fourier time features provide some temporal information, the Transformer relies primarily on learned positional patterns. For long sequences, this may not be sufficient.
- **Impact**: The model may struggle to maintain temporal ordering and causality, especially for sequences with irregular time spacing or long durations.

#### 7. **Overfitting to Training Patterns**
- **Problem**: Transformers are powerful function approximators that can memorize training patterns without generalizing to new physics regimes.
- **Impact**: The model may perform well on training data but fail on validation/test sets with different context parameters or initial conditions.

---

## 4. Direction C: Hybrid PINN

### 4.1 What Changed Compared to Base Model

**Key Architectural Changes:**
- **Combines Direction A and B**: Integrates Transformer encoder (from Direction B) with Latent ODE (from Direction A)
- **Transformer-based z0 encoder**: Uses a Transformer encoder on early time steps to infer initial latent state `z0`, replacing the simple linear z0 encoder
- **Encoder window**: Processes only the first `encoder_window` time steps (default: 10) through the Transformer to compute `z0`
- **Latent ODE evolution**: After computing `z0`, integrates latent ODE over the full time grid (same as Direction A)
- **Deep context encoder** (in C1 variant): Replaces shallow context encoder with a deep MLP (4 layers: 64→128→128→64→32)

**Preserved Components:**
- Same time embedding
- Same loss function structure
- Same input/output interface

**Architecture Details:**
- Transformer encoder: 2 layers, 4 heads (lighter than Direction B)
- Encoder window: 10 time steps (configurable)
- Latent dimension: 64
- Decoder: 3-layer MLP (128 neurons per layer)

### 4.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    └─ context: [batch, 7] ──────────────────┤
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ TimeEmbedding    │
                                    │ (Fourier)        │
                                    └────────┬─────────┘
                                             │
                                             │ t_emb: [batch, N, 17]
                                             │
                                    ┌────────▼─────────┐
                                    │ ContextEncoder   │
                                    │ (Deep in C1)     │
                                    └────────┬─────────┘
                                             │
                                             │ ctx_emb: [batch, N, 64]
                                             │
                                    ┌────────▼─────────┐
                                    │ Concatenate      │
                                    │ [t_emb || ctx]  │
                                    └────────┬─────────┘
                                             │
                                             │ seq_features: [batch, N, 81]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ Transformer Encoder (Window)     │
                                    │ - Process first 10 time steps   │
                                    │ - Self-attention on window      │
                                    │ - Mean pooling → z0_tokens      │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z0_tokens: [batch, 128]
                                             │
                                    ┌────────▼─────────┐
                                    │ z0 Projection    │
                                    │ Linear(128→64)   │
                                    └────────┬─────────┘
                                             │
                                             │ z0: [batch, 64]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ LatentODEBlock                   │
                                    │ (Euler Integration)               │
                                    │ - Integrate over full grid      │
                                    │ - condition = seq_features      │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z_traj: [batch, N, 64]
                                             │
                                    ┌────────▼─────────┐
                                    │ Decoder (MLP)    │
                                    │ 3×128 (tanh)     │
                                    └────────┬─────────┘
                                             │
                                             │ state_pred: [batch, N, 14]
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Loss Function    │
                                    │ (Data+Phys+BC)   │
                                    └──────────────────┘
```

### 4.3 Why It Would Be Better

#### 1. **Best of Both Worlds**
- **Temporal Awareness**: The Transformer encoder on early time steps captures temporal patterns and dependencies, providing a richer initialization for the latent state.
- **Physics Structure**: The latent ODE ensures that state evolution follows continuous dynamics, providing physics consistency.
- **Complementary Strengths**: Combines the pattern recognition of Transformers with the structural guarantees of ODE integration.

#### 2. **Better z0 Initialization**
- **Problem in Direction A**: Simple linear z0 encoder from context only
- **Solution in Direction C**: Transformer processes early time steps to infer `z0`, potentially capturing initial trajectory dynamics
- **Impact**: More accurate initial latent state leads to better trajectory evolution

#### 3. **Efficient Attention**
- **Reduced Complexity**: Only processes `encoder_window` (10) time steps through Transformer, not the full sequence (1501), reducing computational cost
- **Focused Attention**: Early time steps are most critical for trajectory initialization, so focusing attention here is efficient

#### 4. **Deep Context Encoding** (C1 variant)
- **Better Parameter Interactions**: Deep context encoder (4 layers) can capture complex interactions between context parameters
- **Impact**: Better distinction between different physics regimes, leading to more accurate trajectory predictions

### 4.4 Why It's Not Working / Giving Good Results

#### 1. **Complexity and Training Difficulty**
- **Problem**: Combining Transformer and ODE integration creates a complex architecture with many components to train simultaneously:
  - Transformer encoder weights
  - z0 projection weights
  - Latent dynamics network weights
  - Decoder weights
- **Impact**: 
  - Difficult to balance training of all components
  - May require careful initialization and learning rate scheduling
  - Higher risk of training instability

#### 2. **Limited Encoder Window**
- **Problem**: Only the first 10 time steps are used to compute `z0`. For long trajectories (1501 points), this may not capture enough information about the initial trajectory dynamics.
- **Impact**: 
  - If early trajectory is noisy or unrepresentative, `z0` will be poorly initialized
  - May miss important initial conditions that affect the entire trajectory
  - Window size is a hyperparameter that may need careful tuning

#### 3. **Information Bottleneck**
- **Problem**: The Transformer encoder must compress information from 10 time steps (each with 81 features) into a single 128-D vector, which is then projected to 64-D `z0`. This is a significant compression.
- **Impact**: Information loss during compression may lead to inaccurate `z0`, which propagates through ODE integration.

#### 4. **Integration Error Still Present**
- **Problem**: Same integration error accumulation issues as Direction A (Euler method, long trajectories)
- **Impact**: Even with better `z0`, numerical integration errors compound over 1501 time steps

#### 5. **Mismatch Between Transformer and ODE**
- **Problem**: The Transformer learns discrete sequence patterns, while the ODE models continuous dynamics. There may be a mismatch between what the Transformer infers and what the ODE can represent.
- **Impact**: The `z0` computed by the Transformer may not be compatible with the latent dynamics learned by the ODE network, leading to poor trajectory evolution.

#### 6. **Gradient Flow Issues**
- **Problem**: Backpropagating through both Transformer and ODE integration creates a long gradient path:
  ```
  Loss → Decoder → ODE Integration → z0 → Transformer → Input
  ```
- **Impact**: 
  - Gradients may vanish or explode through this long path
  - Difficult to train all components effectively
  - May require gradient clipping or specialized training strategies

#### 7. **Hyperparameter Sensitivity**
- **Problem**: Many hyperparameters to tune:
  - Encoder window size
  - Transformer depth and width
  - Latent dimension
  - ODE dynamics network size
  - Decoder size
- **Impact**: Model performance may be highly sensitive to hyperparameter choices, making it difficult to find good configurations.

#### 8. **No Guaranteed Improvement**
- **Problem**: Combining two architectures doesn't guarantee better performance. The added complexity may introduce new failure modes without solving the original problems.
- **Impact**: May perform worse than simpler Direction A or B models if components don't work well together.

---

## 5. Direction C1: Enhanced Hybrid PINN

### 5.1 What Changed Compared to Direction C

**Key Architectural Enhancements:**

#### 1. **Deep Context Encoder** (Replaces Shallow Encoder)
- **Direction C**: Simple `ContextEncoder` (Linear: 7 → 64, Tanh)
- **Direction C1**: `DeepContextEncoder` with 4 hidden layers:
  ```
  context [7] → Linear(7→64) → GELU → LayerNorm
            → Linear(64→128) → GELU → LayerNorm
            → Linear(128→128) → GELU
            → Linear(128→64) → GELU
            → Linear(64→32) → ctx_emb [32]
  ```
- **Purpose**: Capture complex interactions between context parameters through deeper non-linear transformations

#### 2. **Split Output Heads** (Replaces Single MLP Output)
- **Direction C**: Single decoder MLP outputs 14-D state directly
- **Direction C1**: Three specialized output heads:
  - **Translation Head**: `Linear(128 → 6)` → `[x, y, z, vx, vy, vz]`
  - **Rotation Head**: `Linear(128 → 7)` → `[q0, q1, q2, q3, wx, wy, wz]`
  - **Mass Head**: `Linear(128 → 1)` → `[m]`
- **Purpose**: Allow specialized learning for different state components with different physical properties

#### 3. **Quaternion Normalization** (New Constraint Enforcement)
- **Direction C**: No explicit quaternion constraint
- **Direction C1**: Explicit unit-norm normalization:
  ```python
  quat_raw = rotation_head_output[..., :4]  # [q0, q1, q2, q3]
  quat_norm = normalize_quaternion(quat_raw)  # Enforces ||q|| = 1
  rotation = [quat_norm || ang_vel]
  ```
- **Purpose**: Enforce quaternion unit-norm constraint structurally, preventing invalid rotations

#### 4. **Residual/Delta Prediction** (Replaces Absolute State Prediction)
- **Direction C**: Predicts absolute state `s(t)` directly
- **Direction C1**: Predicts state delta and adds to initial state:
  ```python
  state_delta = [translation_delta, rotation_delta, mass_delta]
  state = initial_state + state_delta
  ```
- **Purpose**: 
  - Easier learning: model only needs to predict changes from known initial condition
  - Better boundary condition satisfaction: `state(t=0) = s0` by construction
  - Reduces prediction range: deltas are typically smaller than absolute states

#### 5. **Initial State Requirement** (New Input)
- **Direction C**: Only requires `(t, context)`
- **Direction C1**: Requires `(t, context, initial_state)` where `initial_state = s(t=0)`
- **Purpose**: Provides ground truth initial condition as baseline for residual prediction

#### 6. **Debug Statistics Tracking** (New Monitoring)
- **Direction C**: No built-in statistics
- **Direction C1**: Tracks:
  - Quaternion norm statistics (raw vs normalized)
  - Mass violation ratio (detects non-physical mass increases)
  - Delta state L2 norms
  - Context embedding statistics
- **Purpose**: Enable monitoring of model behavior and constraint violations during training

**Preserved Components:**
- Same Transformer encoder structure for z0
- Same Latent ODE integration
- Same decoder architecture (but outputs to split heads instead of direct state)
- Same loss function structure

### 5.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    ├─ context: [batch, 7] ─────────────────┤
    └─ initial_state: [batch, 14] ───────────┤ (NEW: s0)
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ TimeEmbedding    │
                                    │ (Fourier)        │
                                    └────────┬─────────┘
                                             │
                                             │ t_emb: [batch, N, 17]
                                             │
                                    ┌────────▼──────────────────────┐
                                    │ DeepContextEncoder          │
                                    │ (4 layers: 7→64→128→128→64→32)│
                                    │ GELU + LayerNorm            │
                                    └────────┬─────────────────────┘
                                             │
                                             │ ctx_emb: [batch, N, 32]
                                             │
                                    ┌────────▼─────────┐
                                    │ Concatenate      │
                                    │ [t_emb || ctx]  │
                                    └────────┬─────────┘
                                             │
                                             │ seq_features: [batch, N, 49]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ Transformer Encoder (Window)      │
                                    │ - Process first 10 time steps    │
                                    │ - Mean pooling → z0_tokens      │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z0: [batch, 64]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ LatentODEBlock                    │
                                    │ (Euler Integration)               │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z_traj: [batch, N, 64]
                                             │
                                    ┌────────▼─────────┐
                                    │ Decoder (MLP)    │
                                    │ 3×128 (tanh)     │
                                    └────────┬─────────┘
                                             │
                                             │ decoder_features: [batch, N, 128]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ OutputHeads (Split)               │
                                    │ - Translation: [batch, N, 6]      │
                                    │ - Rotation: [batch, N, 7]         │
                                    │ - Mass: [batch, N, 1]            │
                                    └────────┬─────────────────────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Quaternion Norm  │
                                    │ normalize_quat() │
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Concatenate      │
                                    │ [trans || rot_norm || mass]│
                                    └────────┬─────────┘
                                             │
                                             │ state_delta: [batch, N, 14]
                                             │
                                    ┌────────▼─────────┐
                                    │ Add Initial State │
                                    │ s0: [batch, 14]  │
                                    │ → broadcast to [batch, N, 14]│
                                    └────────┬─────────┘
                                             │
                                             │ state: [batch, N, 14] = s0 + delta
                                             │
                                    ┌────────▼─────────┐
                                    │ Debug Stats      │
                                    │ (quat, mass, etc)│
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Loss Function    │
                                    │ (Data+Phys+BC)   │
                                    └──────────────────┘
```

### 5.3 Why C1 Features Make a Difference Compared to Direction C

#### 1. **Deep Context Encoder: Better Physics Regime Distinction**

**Why It Matters:**
- **Direction C Problem**: Shallow encoder (7 → 64) cannot capture complex parameter interactions. For example:
  - High `wind_mag` + low `Cd` → different trajectory than low `wind_mag` + high `Cd`
  - `Tmax` and `m0` interact non-linearly to determine acceleration profile
  - `CL_alpha` and `Cm_alpha` together determine aerodynamic coupling
- **C1 Solution**: Deep encoder with 4 hidden layers and GELU activations can learn:
  - Multi-level feature hierarchies
  - Non-linear parameter interactions
  - Context-dependent embeddings that better distinguish physics regimes
- **Expected Impact**: 
  - More accurate `z0` initialization (better context → z0 mapping)
  - Better conditioning of latent dynamics (context embedding affects ODE evolution)
  - Improved generalization to unseen context parameter combinations

**Specific Improvement**: The deep encoder transforms context through multiple non-linear layers, allowing it to learn that `[high wind_mag, low Cd]` should produce a different embedding than `[low wind_mag, high Cd]`, even if they might produce similar intermediate states. This distinction propagates through the Transformer encoder and affects the entire trajectory evolution.

#### 2. **Split Output Heads: Specialized Learning for Different State Components**

**Why It Matters:**
- **Direction C Problem**: Single MLP must learn to predict all 14 state components simultaneously, despite them having very different:
  - Physical properties (position vs. velocity vs. quaternion vs. mass)
  - Scales and ranges
  - Dynamics characteristics
- **C1 Solution**: Three specialized heads allow:
  - **Translation head**: Focuses on learning position/velocity dynamics (6-D, continuous, unbounded)
  - **Rotation head**: Focuses on quaternion + angular velocity (7-D, quaternion has unit-norm constraint)
  - **Mass head**: Focuses on monotonic decrease (1-D, always decreasing)
- **Expected Impact**:
  - Each head can learn component-specific patterns
  - Better handling of quaternion constraints (rotation head can specialize)
  - More stable training (specialized heads reduce interference between components)

**Specific Improvement**: The rotation head can learn that quaternions should be normalized and that angular velocity couples with orientation. The mass head can learn that mass should only decrease. The translation head can focus on smooth position/velocity trajectories. This specialization prevents the model from trying to learn conflicting patterns in a single output layer.

#### 3. **Quaternion Normalization: Structural Constraint Enforcement**

**Why It Matters:**
- **Direction C Problem**: Quaternions predicted by MLP may not satisfy unit-norm constraint `||q|| = 1`. This leads to:
  - Invalid rotations (non-unit quaternions)
  - Physics violations (dynamics assume unit quaternions)
  - Numerical instability in quaternion operations
- **C1 Solution**: Explicit normalization after rotation head output:
  ```python
  quat_norm = q / ||q||  # Enforces unit norm
  ```
- **Expected Impact**:
  - Guarantees valid quaternions at every time step
  - Reduces physics loss (dynamics computed with valid quaternions)
  - Improves numerical stability
  - Better orientation predictions

**Specific Improvement**: By normalizing quaternions structurally, the model doesn't need to learn the constraint through loss penalties. This is especially important because:
- Quaternion dynamics are highly sensitive to norm violations
- The physics loss `L_phys` assumes unit quaternions
- Invalid quaternions cause cascading errors in rotation-dependent forces (aerodynamics, gravity)

#### 4. **Residual/Delta Prediction: Easier Learning and Boundary Satisfaction**

**Why It Matters:**
- **Direction C Problem**: Predicting absolute state `s(t)` directly is difficult because:
  - Must learn entire state space (large output range)
  - Boundary condition `s(t=0) = s0` must be learned through loss
  - Small errors in absolute prediction can be large relative to true state
- **C1 Solution**: Predict delta `Δs(t) = s(t) - s0` and add to known initial state:
  ```python
  state = s0 + state_delta  # Boundary condition satisfied by construction
  ```
- **Expected Impact**:
  - **Easier learning**: Model only predicts changes, which are typically smaller
  - **Automatic boundary satisfaction**: `state(t=0) = s0 + 0 = s0` (if delta at t=0 is small)
  - **Better numerical stability**: Deltas are better scaled than absolute states
  - **Reduced boundary loss**: Less need for explicit boundary loss penalty

**Specific Improvement**: Instead of learning "predict position x(t) = 10000m", the model learns "predict position change Δx(t) = 50m from initial". This is much easier because:
- Deltas are typically smaller in magnitude
- The model can focus on learning dynamics (how state changes) rather than absolute values
- Initial condition is guaranteed correct (if delta prediction is accurate at t=0)

#### 5. **Initial State Input: Stronger Baseline and Better Conditioning**

**Why It Matters:**
- **Direction C Problem**: Model must infer initial state from context only, which is difficult:
  - Context parameters don't directly specify initial state
  - Initial state depends on launch conditions not in context
  - Poor initial state leads to poor entire trajectory
- **C1 Solution**: Initial state provided as input, used as baseline:
  ```python
  state = initial_state + predicted_delta
  ```
- **Expected Impact**:
  - Model doesn't need to learn initial state from context
  - Can focus on learning trajectory evolution
  - Better conditioning: accurate starting point for ODE integration
  - More stable training: less uncertainty about initial conditions

**Specific Improvement**: By providing the true initial state, the model can:
- Start ODE integration from the correct point
- Learn trajectory evolution without worrying about initial state inference
- Reduce the burden on the Transformer encoder (doesn't need to infer s0 from context)

#### 6. **Debug Statistics: Monitoring and Diagnostics**

**Why It Matters:**
- **Direction C Problem**: No visibility into model behavior during training:
  - Can't detect quaternion norm violations
  - Can't monitor mass physics violations
  - Difficult to diagnose training issues
- **C1 Solution**: Tracks key statistics:
  - Quaternion norms (detect normalization issues)
  - Mass violations (detect non-physical mass increases)
  - Delta state norms (monitor prediction magnitude)
  - Context embedding stats (monitor encoder behavior)
- **Expected Impact**:
  - Early detection of constraint violations
  - Better understanding of model behavior
  - Easier debugging and hyperparameter tuning
  - Validation of architectural choices

**Specific Improvement**: Debug stats enable:
- Real-time monitoring: detect if quaternion normalization is working
- Physics validation: catch mass increase violations early
- Training diagnostics: understand if model is learning correctly
- Ablation studies: compare with/without features

### 5.4 Why C1 May Still Not Be Working / Giving Good Results

#### 1. **Residual Prediction Limitations**
- **Problem**: While residual prediction helps with boundary conditions, it may introduce new issues:
  - Model must learn to predict zero delta at t=0, which may be difficult
  - If initial state is noisy or inaccurate, residual prediction amplifies errors
  - Delta prediction may not capture large state changes well
- **Impact**: Model may struggle with trajectories that have large deviations from initial state, or may produce small deltas even when large changes are needed.

#### 2. **Deep Context Encoder Overfitting**
- **Problem**: Deep encoder (4 layers) with many parameters may:
  - Overfit to training context parameter distributions
  - Learn spurious correlations between context parameters
  - Fail to generalize to unseen context combinations
- **Impact**: Model may perform well on training data but fail on validation/test sets with different context parameters.

#### 3. **Split Heads Coordination Issues**
- **Problem**: Three separate heads must coordinate to produce consistent state:
  - Translation, rotation, and mass predictions must be physically consistent
  - No explicit mechanism ensures heads produce compatible outputs
  - May learn conflicting patterns (e.g., translation suggests high velocity but rotation suggests high drag)
- **Impact**: Individual components may be accurate but overall trajectory may be inconsistent or unphysical.

#### 4. **Quaternion Normalization Gradient Issues**
- **Problem**: Normalization is a non-differentiable operation (division by norm):
  - Gradients through normalization can be unstable
  - Normalization may mask underlying quaternion prediction issues
  - Model may learn to predict unnormalized quaternions and rely on normalization to fix them
- **Impact**: Quaternion predictions may be poor, but normalization hides the problem until it causes issues in downstream computations.

#### 5. **Initial State Dependency**
- **Problem**: Model requires true initial state, which may not always be available:
  - In real applications, initial state may be uncertain or noisy
  - Model cannot be used for prediction without known initial state
  - Defeats purpose of learning from context if initial state must be provided
- **Impact**: Limited applicability to scenarios where initial state is unknown or must be inferred.

#### 6. **Complexity and Training Difficulty**
- **Problem**: C1 adds significant complexity:
  - Deep context encoder (more parameters)
  - Split output heads (more output layers)
  - Normalization operations (additional computations)
  - Debug statistics (overhead)
- **Impact**: 
  - Harder to train (more components to optimize)
  - More hyperparameters to tune
  - Longer training time
  - Higher risk of training instability

#### 7. **Integration Error Still Present**
- **Problem**: Same ODE integration issues as Direction C:
  - Euler method accumulates errors
  - Long trajectories (1501 points) compound integration errors
  - No improvement in numerical integration method
- **Impact**: Even with better architecture, integration errors may dominate prediction errors.

#### 8. **Limited Encoder Window**
- **Problem**: Still only uses first 10 time steps for z0 computation:
  - May not capture enough initial trajectory information
  - Window size is still a hyperparameter
  - Information bottleneck remains
- **Impact**: Better context encoding doesn't solve the limited window problem.

#### 9. **No Physics in Latent Dynamics**
- **Problem**: Latent dynamics network still learns purely from data:
  - No built-in knowledge of physical dynamics
  - Must learn entire 6-DOF dynamics from scratch
  - May learn unphysical latent dynamics
- **Impact**: Architecture improvements don't address the fundamental challenge of learning complex physics.

#### 10. **Debug Stats Don't Fix Problems**
- **Problem**: Debug statistics only monitor issues, they don't fix them:
  - Can detect quaternion violations but normalization may not solve underlying problem
  - Can detect mass violations but no mechanism to prevent them
  - Statistics are diagnostic, not corrective
- **Impact**: Better visibility into problems doesn't necessarily lead to better solutions.

---

## 6. Direction C2: Shared Stem + Dedicated Branches

### 6.1 What Changed Compared to Direction C1

**Key Architectural Changes:**

#### 1. **Shared Stem Module** (Replaces Separate Processing)
- **Direction C1**: Time embedding and context encoding processed separately, then concatenated before Transformer encoder window
- **Direction C2**: New `SharedStem` module that:
  - Processes time + context together through unified pipeline
  - Applies temporal modeling (Transformer) to the **entire sequence** (not just encoder window)
  - Produces shared embedding `[batch, N, hidden_dim]` that captures global physics patterns
- **Purpose**: Learn unified temporal + context representation that all branches can leverage

#### 2. **Dedicated Branches Architecture** (Replaces Split Output Heads)
- **Direction C1**: Split output heads (TranslationHead, RotationHead, MassHead) that process decoder features independently
- **Direction C2**: Three completely independent branch networks:
  - **TranslationBranch**: Specialized MLP `[latent_dim → 128 → 128 → 6]` for `[x, y, z, vx, vy, vz]`
  - **RotationBranch**: Specialized MLP `[latent_dim → 256 → 256 → 7]` for `[q0, q1, q2, q3, wx, wy, wz]`
  - **MassBranch**: Specialized MLP `[latent_dim → 64 → 1]` for `[Δm]`
- **Key Difference**: Branches receive **shared embedding from Latent ODE** (not decoder features), enabling them to learn from global physics representation

#### 3. **Full-Sequence Temporal Modeling** (Replaces Window-Only Processing)
- **Direction C1**: Transformer encoder processes only first 10 time steps (encoder window) to compute z0
- **Direction C2**: Shared Stem processes **entire time sequence** (N=1501) through Transformer before deriving z0
- **Impact**: All time steps contribute to shared representation, not just initial window

#### 4. **Shared Embedding Flow**
- **Direction C1**: Flow: `(t, context) → z0 → ODE → decoder → split heads`
- **Direction C2**: Flow: `(t, context) → Shared Stem → shared_emb → z0 → ODE → z_traj → dedicated branches`
- **Key Innovation**: Shared Stem learns global physics, then branches specialize on specific subsystems

**Preserved Components:**
- Same DeepContextEncoder (Set#2) - now part of Shared Stem
- Same Latent ODE evolution (from Direction C)
- Same quaternion normalization (Set#1)
- Same Δ-state reconstruction (Set#1)
- Same loss function structure
- Same input/output interface: `(t, context, initial_state) → state`

**Architecture Details:**
- Shared Stem: Transformer encoder (4 layers, 4 heads, 128D hidden)
- Shared embedding dimension: 128 (configurable)
- Translation branch: 2 hidden layers (128 neurons each)
- Rotation branch: 2 hidden layers (256 neurons each, wider for complexity)
- Mass branch: 1 hidden layer (64 neurons, narrower for simplicity)
- Encoder window: 10 time steps (for z0 derivation from shared embedding)

### 6.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    ├─ context: [batch, 7] ─────────────────┤
    └─ initial_state: [batch, 14] ──────────┤
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Shared Stem      │
                                    │                  │
                                    │ 1. TimeEmbedding │
                                    │    (Fourier)     │
                                    │    → [batch,N,17]│
                                    │                  │
                                    │ 2. DeepContext   │
                                    │    Encoder       │
                                    │    → [batch,N,32]│
                                    │                  │
                                    │ 3. Concatenate   │
                                    │    [t_emb||ctx]  │
                                    │    → [batch,N,49]│
                                    │                  │
                                    │ 4. Input Proj    │
                                    │    Linear(49→128)│
                                    │    → [batch,N,128]│
                                    │                  │
                                    │ 5. Transformer   │
                                    │    Encoder       │
                                    │    (4 layers)    │
                                    │    → [batch,N,128]│
                                    └────────┬─────────┘
                                             │
                                             │ shared_emb: [batch, N, 128]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ z0 Derivation                    │
                                    │ - Take first 10 time steps      │
                                    │ - Mean pooling → z0_tokens      │
                                    │ - Project to latent_dim         │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z0: [batch, 64]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ LatentODEBlock                    │
                                    │ (Euler Integration)               │
                                    │ - condition = shared_emb         │
                                    │ - Integrate over full grid      │
                                    └────────┬─────────────────────────┘
                                             │
                                             │ z_traj: [batch, N, 64]
                                             │
                                    ┌────────▼──────────────────────────┐
                                    │ Dedicated Branches                │
                                    │                                   │
                                    │ ┌──────────┬──────────┬─────────┐│
                                    │ │Trans     │Rotation  │Mass     ││
                                    │ │Branch    │Branch    │Branch   ││
                                    │ │[64→128→  │[64→256→  │[64→64→  ││
                                    │ │128→6]    │256→7]    │1]      ││
                                    │ └────┬─────┴────┬─────┴────┬────┘│
                                    │      │          │          │      │
                                    │      │          │          │      │
                                    │ trans:│rotation:│mass:     │      │
                                    │[batch,│[batch,  │[batch,   │      │
                                    │N,6]   │N,7]     │N,1]      │      │
                                    └───────┴──────────┴──────────┴──────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Quaternion Norm  │
                                    │ normalize_quat() │
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Concatenate      │
                                    │ [trans||rot_norm||mass]│
                                    └────────┬─────────┘
                                             │
                                             │ state_delta: [batch, N, 14]
                                             │
                                    ┌────────▼─────────┐
                                    │ Add Initial State │
                                    │ s0: [batch, 14]  │
                                    │ → broadcast      │
                                    └────────┬─────────┘
                                             │
                                             │ state: [batch, N, 14] = s0 + delta
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Loss Function    │
                                    │ (Data+Phys+BC)   │
                                    └──────────────────┘
```

### 6.3 Why It Would Be Better

#### 1. **Shared Learning of Global Physics**

**Why It Matters:**
- **Direction C1 Problem**: Split heads process decoder features independently, but decoder features may not capture global physics patterns effectively. Each head must learn physics relationships from its own perspective.
- **C2 Solution**: Shared Stem processes entire sequence through Transformer, learning global temporal + context patterns. All branches receive the same shared representation, enabling them to leverage global physics knowledge.
- **Expected Impact**:
  - Branches can learn from global patterns (e.g., "high altitude → low density → different aerodynamics")
  - Shared representation captures cross-component relationships (e.g., rotation affects translation through aerodynamic forces)
  - Better generalization: shared physics knowledge transfers across branches

**Specific Improvement**: The Shared Stem learns that "early in trajectory, high thrust + low mass → high acceleration" and "late in trajectory, low mass + high altitude → different dynamics". This global knowledge is available to all branches, so:
- Translation branch knows when to expect high velocities
- Rotation branch knows when aerodynamic forces are significant
- Mass branch knows the depletion profile affects all other components

#### 2. **True Specialization Without Interference**

**Why It Matters:**
- **Direction C1 Problem**: Split heads share the same decoder features, which may cause interference. For example, if decoder learns a pattern that helps translation but hurts rotation, both heads are affected.
- **C2 Solution**: Dedicated branches are completely independent networks with separate parameters. Each branch can specialize without affecting others.
- **Expected Impact**:
  - Translation branch can learn smooth position/velocity patterns without worrying about quaternion constraints
  - Rotation branch can focus on unit-norm quaternions and angular velocity coupling
  - Mass branch can learn monotonic decrease without interference from other components
  - No gradient conflicts between branches

**Specific Improvement**: The rotation branch (256 neurons) can dedicate all its capacity to learning quaternion dynamics and angular velocity, while the translation branch (128 neurons) focuses on simpler position/velocity patterns. The mass branch (64 neurons) handles the simplest task. This specialization prevents the model from trying to learn conflicting patterns in a shared representation.

#### 3. **Full-Sequence Temporal Awareness**

**Why It Matters:**
- **Direction C1 Problem**: Transformer encoder processes only first 10 time steps for z0 computation. The rest of the sequence is not processed through Transformer, missing long-range temporal dependencies.
- **C2 Solution**: Shared Stem processes **entire sequence** (N=1501) through Transformer before deriving z0. All time steps contribute to shared representation.
- **Expected Impact**:
  - Captures long-range dependencies (e.g., early mass depletion affects late trajectory)
  - Better temporal pattern recognition across full trajectory
  - More accurate z0 initialization (derived from full-sequence understanding)
  - Improved physics consistency (global patterns inform local evolution)

**Specific Improvement**: The Shared Stem can learn that "if mass depletes quickly in first 100 steps, then late trajectory will have different dynamics". This global understanding informs the z0 initialization and affects the entire ODE evolution, leading to more physically consistent trajectories.

#### 4. **Hierarchical Feature Learning**

**Why It Matters:**
- **Direction C1 Problem**: Features flow: `z0 → ODE → decoder → heads`. This is a linear pipeline where each stage must learn everything from scratch.
- **C2 Solution**: Hierarchical learning:
  - **Level 1 (Shared Stem)**: Learns global physics patterns (temporal + context)
  - **Level 2 (Latent ODE)**: Learns state evolution dynamics
  - **Level 3 (Branches)**: Learns component-specific mappings
- **Expected Impact**:
  - Each level can focus on its specific task
  - Lower levels provide rich features for higher levels
  - Better feature reuse and efficiency

**Specific Improvement**: The Shared Stem learns "high wind_mag + low Cd → different trajectory shape", which is encoded in the shared embedding. The Latent ODE uses this embedding to evolve state, and the branches decode to specific components. This hierarchical structure is more efficient than learning everything end-to-end.

#### 5. **Better Branch Capacity Allocation**

**Why It Matters:**
- **Direction C1 Problem**: Split heads have same architecture (same width), but different components have different complexity:
  - Rotation (quaternion + angular velocity) is most complex
  - Translation (position + velocity) is moderate
  - Mass (single scalar) is simplest
- **C2 Solution**: Branches have different capacities:
  - Rotation branch: 256 neurons (wider, more complex)
  - Translation branch: 128 neurons (moderate)
  - Mass branch: 64 neurons (narrower, simpler)
- **Expected Impact**:
  - More parameters where needed (rotation)
  - Fewer parameters where not needed (mass)
  - Better parameter efficiency
  - Improved learning for complex components

**Specific Improvement**: The rotation branch gets 256 neurons to handle:
- Quaternion unit-norm constraint
- Angular velocity dynamics
- Coupling with translation through aerodynamics
- Complex rotation patterns

Meanwhile, the mass branch only needs 64 neurons to learn monotonic decrease, which is much simpler.

#### 6. **Maintains All C1 Advantages**

**Why It Matters:**
- **C2 Preserves**: All improvements from C1:
  - Deep context encoder (Set#2)
  - Quaternion normalization (Set#1)
  - Δ-state reconstruction (Set#1)
  - Debug statistics
- **C2 Adds**: Shared Stem + Dedicated Branches on top
- **Expected Impact**: Gets all C1 benefits plus new architectural improvements

**Specific Improvement**: C2 is not a replacement but an evolution. It keeps what works (deep context, quaternion norm, Δ-state) and adds new capabilities (shared learning, true specialization, full-sequence awareness).

### 6.4 Why It May Still Not Work / Give Good Results

#### 1. **Shared Stem May Learn Spurious Patterns**

**Problem**: The Shared Stem processes entire sequence through Transformer, which may learn patterns that don't correspond to physical dynamics:
- Attention may focus on spurious correlations
- Global patterns may not generalize to unseen contexts
- Transformer may overfit to training sequence structures

**Impact**: If Shared Stem learns wrong patterns, all branches receive incorrect shared representation, leading to poor predictions across all components.

#### 2. **Information Bottleneck in Shared Embedding**

**Problem**: Shared Stem must compress information from:
- Time features (17D)
- Context embedding (32D)
- Temporal patterns (N=1501 time steps)

Into a single 128D shared embedding that all branches use. This is a significant compression that may lose important information.

**Impact**: 
- Critical physics information may be lost in compression
- Branches may receive insufficient information for accurate predictions
- Different branches may need different information, but they all get the same shared embedding

#### 3. **Branch Independence May Prevent Coordination**

**Problem**: While branch independence prevents interference, it also prevents explicit coordination:
- Translation, rotation, and mass predictions must be physically consistent
- No mechanism ensures branches produce compatible outputs
- May learn conflicting patterns (e.g., translation suggests high velocity but rotation suggests high drag)

**Impact**: Individual components may be accurate but overall trajectory may be inconsistent or unphysical.

#### 4. **Full-Sequence Transformer Complexity**

**Problem**: Processing entire sequence (N=1501) through Transformer is computationally expensive:
- O(N²) attention complexity
- Memory-intensive for long sequences
- May require gradient checkpointing or sequence truncation

**Impact**:
- Slow training
- Limited batch size
- May not scale to longer trajectories

#### 5. **z0 Derivation Still Limited**

**Problem**: Even though Shared Stem processes full sequence, z0 is still derived from only first 10 time steps (encoder window):
- Information bottleneck remains
- May not capture enough initial trajectory dynamics
- Window size is still a hyperparameter

**Impact**: Better shared representation doesn't solve the limited window problem for z0 initialization.

#### 6. **Integration Error Still Present**

**Problem**: Same ODE integration issues as previous directions:
- Euler method accumulates errors
- Long trajectories (1501 points) compound integration errors
- No improvement in numerical integration method

**Impact**: Even with better architecture, integration errors may dominate prediction errors.

#### 7. **No Physics in Latent Dynamics**

**Problem**: Latent dynamics network still learns purely from data:
- No built-in knowledge of physical dynamics
- Must learn entire 6-DOF dynamics from scratch
- May learn unphysical latent dynamics

**Impact**: Architecture improvements don't address the fundamental challenge of learning complex physics.

#### 8. **Hyperparameter Sensitivity**

**Problem**: Many hyperparameters to tune:
- Shared Stem depth and width
- Branch architectures (translation/rotation/mass dimensions)
- Encoder window size
- Latent dimension
- ODE dynamics network size

**Impact**: Model performance may be highly sensitive to hyperparameter choices, making it difficult to find good configurations.

#### 9. **Training Complexity**

**Problem**: C2 adds significant complexity:
- Shared Stem (Transformer with 4 layers)
- Three independent branches
- Latent ODE integration
- Multiple loss components

**Impact**:
- Harder to train (more components to optimize)
- Longer training time
- Higher risk of training instability
- May require careful initialization and learning rate scheduling

#### 10. **Limited Improvement Over C1**

**Problem**: C2 may not provide significant improvement over C1:
- C1 already has deep context encoder, split heads, quaternion norm, Δ-state
- C2 adds Shared Stem and dedicated branches, but may not solve fundamental issues
- Added complexity may not be worth the marginal improvement

**Impact**: May perform similarly to C1 but with more complexity and training difficulty.

---

## 6.5 Direction C3: Enhanced Hybrid PINN with RMSE Reduction Solutions

**Date**: 2025-XX-XX (Planned)  
**Base Architecture**: C2 (Shared Stem + Dedicated Branches)  
**Purpose**: Address RMSE root causes through architectural improvements

### 6.5.1 Motivation

**C2 Baseline Performance (exp3)**:
- Total RMSE: **0.96**
- Translation RMSE: 1.41
- Rotation RMSE: **0.38** (3.5x worse than exp2)
- Mass RMSE: 0.19
- **Key Issues**: Quaternion norm=1.08, high vertical dynamics errors (z: 0.91-1.10, vz: 2.98-3.45), mass violations: 4.2%

**Failed Approach (exp4 - Loss Weighting)**:
- Total RMSE: **1.005** (worse than C2!)
- Mass violations: **4.2%** (physically impossible)
- **Conclusion**: Loss weighting doesn't fix root causes, only penalizes errors

**C3 Solution**: Architectural improvements that address root causes structurally, not through penalties.

### 6.5.2 What Changed Compared to Direction C2

**Six Architectural Solutions**:

1. **Solution 1: Physics-Informed Vertical Dynamics Branch**
   - **C2 Problem**: Model must learn `rho(z)` from data, complex altitude→density→drag chain not explicit
   - **C3 Solution**: `PhysicsAwareTranslationBranch` with explicit physics computation
     - `rho(z) = rho0 * exp(-z/H)` computed directly
     - `drag = 0.5 * rho * |v|² * Cd * S` computed explicitly
     - `vz_corrected = vz - drag_z / m` physics-aware correction
   - **Expected Impact**: z: 0.91-1.10 → 0.60-0.80, vz: 2.98-3.45 → 1.80-2.40

2. **Solution 2: Quaternion Minimal Representation**
   - **C2 Problem**: Normalization `q / ||q||` is non-differentiable, masking effect
   - **C3 Solution**: `RotationBranchMinimal` uses rotation vector (3D) → quaternion conversion
     - `rotation_vector_to_quaternion()`: axis-angle → quaternion (always unit norm)
     - No normalization needed, always produces valid quaternions
   - **Expected Impact**: Rotation RMSE: 0.38 → 0.15-0.25, quaternion norm: 1.08 → 1.0

3. **Solution 3: Structural Mass Monotonicity**
   - **C2 Problem**: Mass branch predicts independently, no `m(t+1) <= m(t)` guarantee (4.2% violations)
   - **C3 Solution**: `MonotonicMassBranch` with structural constraint
     - `mass_delta = -ReLU(mass_delta_raw)` (always ≤ 0)
     - `mass = cumsum(mass_delta) + m0` (always decreasing)
   - **Expected Impact**: Mass violations: 4.2% → 0%, Mass RMSE: 0.19 → 0.10-0.12

4. **Solution 4: Higher-Order ODE Integration (RK4)**
   - **C2 Problem**: Euler method O(dt) error accumulates over 1501 steps
   - **C3 Solution**: `LatentODEBlockRK4` with 4th-order Runge-Kutta
     - Error: O(dt⁴) vs Euler's O(dt) - ~1000x smaller per step
     - 4 function evaluations per step vs Euler's 1
   - **Expected Impact**: Better vertical RMSE, improved long-term stability

5. **Solution 5: Cross-Branch Coordination**
   - **C2 Problem**: Independent branches, no aerodynamic coupling
   - **C3 Solution**: `CoordinatedBranches` with `AerodynamicCouplingModule`
     - Extracts: `|v|`, `q`, `rho`
     - Computes: `drag_correction = f(|v|, q, rho, context)`
     - Applies: `translation += drag_correction`
   - **Expected Impact**: Rotation RMSE: 0.38 → 0.25-0.30, Translation RMSE: 1.41 → 1.10-1.30

6. **Solution 6: Enhanced z0 Initialization**
   - **C2 Problem**: Limited encoder window (10 steps = 6.7% of trajectory)
   - **C3 Solution**: `EnhancedZ0Derivation` with hybrid approach
     - Full sequence mean + window Transformer (data-driven)
     - Physics-informed encoder (physics-based)
     - Blend: `z0 = 0.3*z0_physics + 0.7*z0_data`
   - **Expected Impact**: 5-10% reduction across all RMSE components, faster convergence

**Preserved Components**:
- Same Shared Stem architecture
- Same Latent ODE structure (but with RK4 integration)
- Same input/output interface: `(t, context, initial_state) → state`

### 6.5.3 Architecture Flow

```
Inputs: (t, context, initial_state)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ SHARED STEM (Unchanged from C2)                            │
│ TimeEmbedding + DeepContextEncoder + TransformerEncoder   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ ENHANCED z0 DERIVATION (Solution 6)                         │
│ Hybrid: 30% physics + 70% data                             │
└──────────────┬──────────────────────────────────────────────┘
               │ z0
               ▼
┌─────────────────────────────────────────────────────────────┐
│ RK4 LATENT ODE (Solution 4)                                 │
│ O(dt⁴) error vs Euler's O(dt)                              │
└──────────────┬──────────────────────────────────────────────┘
               │ z_traj
               ▼
┌─────────────────────────────────────────────────────────────┐
│ COORDINATED BRANCHES (Solutions 1,2,3,5)                   │
│                                                              │
│ PhysicsAwareTranslationBranch (Sol 1) → translation        │
│ RotationBranchMinimal (Sol 2) → rotation (unit quat)      │
│ MonotonicMassBranch (Sol 3) → mass (decreasing)            │
│ AerodynamicCouplingModule (Sol 5) → drag corrections       │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
          Δ-state reconstruction
          state = initial_state + state_delta
```

### 6.5.4 Expected Performance

| Component | C2 (exp3) | C3 (Expected) | Improvement |
|-----------|-----------|---------------|-------------|
| **Total RMSE** | 0.96 | **0.60-0.75** | 25-40% |
| **Translation RMSE** | 1.41 | **0.90-1.20** | 15-30% |
| **Rotation RMSE** | 0.38 | **0.15-0.25** | 35-60% |
| **Mass RMSE** | 0.19 | **0.10-0.12** | 20-30% |
| **Mass Violations** | 4.2% | **0%** | 100% fix |
| **Quaternion Norm** | 1.08 | **1.0** | 100% fix |

### 6.5.5 Why It Would Be Better

1. **Structural Constraints**: Mass monotonicity and quaternion unit norm guaranteed by architecture, not loss penalties
2. **Explicit Physics**: Density and drag computed directly, reducing information bottleneck
3. **Higher-Order Integration**: RK4 reduces integration error accumulation significantly
4. **Cross-Branch Coordination**: Aerodynamic coupling addresses rotation-translation coupling errors
5. **Better Initialization**: Hybrid z0 reduces error propagation through ODE

### 6.5.6 Potential Drawbacks

1. **Increased Complexity**: 6 architectural solutions add significant complexity
2. **Computational Cost**: RK4 requires 4x function evaluations vs Euler
3. **Implementation Effort**: Requires implementing multiple new modules
4. **Hyperparameter Tuning**: More parameters to tune (z0 blend, physics layer weights)
5. **Training Stability**: More components may require careful initialization

### 6.5.7 Implementation Status

- ⚠️ **Planned**: C3 architecture designed but not yet implemented
- 📋 **Reference**: See `docs/expANAL_SOLS.md` for detailed implementation guide
- 🔄 **Next Steps**: Implement solutions in phases (high → medium → low priority)

**Implementation Phases**:
1. **Phase 1 (High Priority)**: Solutions 2, 3, 4 (quaternion, mass, RK4)
2. **Phase 2 (Medium Priority)**: Solutions 1, 5 (physics-aware translation, coordination)
3. **Phase 3 (Low Priority)**: Solution 6 (enhanced z0)

---

## 7. Direction D: Dependency-Aware Backbone + Causal Heads

Direction D abandons the latent-ODE + Δ-state stack and instead enforces physics couplings through the order in which outputs are generated: **mass → attitude → translation**. A single shared backbone emits a latent feature that every head can read, but each downstream head also receives the predictions of the upstream head, making the data flow itself respect thrust-to-weight and aerodynamic dependencies. Direction D1 extends the same idea with explicit physics features and causal temporal integration.

### 7.1 What Changed Compared to Direction C2

1. **Single Backbone, No ODE Block** – Time (8 Fourier frequencies → 17D) and context (32D embedding) feed a `[256, 256, 256, 256]` MLP backbone. There is no Transformer, z₀ derivation, or Δ-state reconstruction, so the model does not require `initial_state`.
2. **Dependency-Preserving Heads (G3→G2→G1)** – The mass head (G3) fires first and its output is concatenated into the inputs of the attitude head (G2), which then feeds the translation/acceleration head (G1). Translation therefore cannot contradict the mass/attitude it depends on.
3. **Physics & Causality Hooks (Direction D1)** – D1 swaps quaternion output for a 6D rotation representation, injects density/dynamic-pressure/aero coefficients from `PhysicsComputationLayer`, predicts acceleration instead of Δ-state, and reconstructs velocity/position through an RK4 `TemporalIntegrator`.
4. **Lightweight Training** – Pure MLP stack with GELU + LayerNorm keeps wall-clock time ≈40% lower than C2 while still supporting the same loss structure (data + physics + boundary).

### 7.2 Architecture Outline

```
t → FourierFeatures (17) ----┐
context → ContextEncoder ----┤
                              ▼
                        Shared Backbone
                           [256×4]
                              │ latent
                              ▼
                 ┌────────────────────────┐
                 │ G3: Mass Head          │ → m_pred
                 └──────────┬─────────────┘
                            ▼
                 ┌────────────────────────┐
                 │ G2: Attitude + ω Head  │
                 │ input = [latent || m]  │ → q_pred, w_pred
                 └──────────┬─────────────┘
                            ▼
                 ┌────────────────────────────────┐
                 │ G1: Translation / Accel Head   │
                 │ input = [latent || m || q || w │
                 │          (+ physics in D1)]     │
                 └──────────┬──────────────────────┘
                            ▼
                 ┌────────────────────────────────────────┐
                 │ Direction D: concat → state [14]       │
                 │ Direction D1: RK4 integrate accel → v,z│
                 └────────────────────────────────────────┘
```

### 7.3 Feature Highlights

- **Ordered Coupling Instead of Auxiliary Losses** – Translation is conditioned on the mass and quaternion it depends on, so thrust-to-weight and aerodynamic coupling emerge from the architecture itself.
- **Shared Context Without Attention** – The backbone embeds global scenario information without a Transformer, keeping gradients stable and inference latency low.
- **Quaternion Stability** – D normalizes quaternions immediately after G2; D1’s 6D representation removes normalization issues entirely.
- **Physics Hints (D1)** – Density, dynamic pressure, and aerodynamic coefficients guide the attitude and acceleration heads toward physically plausible regimes.
- **Causal Integration** – D1 integrates acceleration with RK4, enforcing that velocity/position are time integrals of the predicted accelerations.

### 7.4 Early Results (exp6 & exp7)

| Experiment | Model | Total RMSE | Translation RMSE | Rotation RMSE | Notes |
|------------|-------|------------|------------------|---------------|-------|
| `exp6_24_11_direction_d_baseline` | Direction D | **0.300** | 0.436 | 0.127 | Fastest training loop; quaternions stay on-unit after G2 normalization. |
| `exp7_24_11_direction_d1_baseline` | Direction D1 | **0.285** | 0.382 | 0.188 | RK4 integrator lowers `vz` error but 6D→quat decoding still tuning-sensitive. |

**Observations**
- Mass predictions remain monotonic without explicit Δ-state constraints, but Δ magnitude (mean ≈ 2.5) is higher than Δ-state models. Light slope regularization could help.
- Physics-aware features reduce vertical acceleration error yet introduce sensitivity to aero coefficients; we may need to regularize the physics layer outputs.
- Removing `initial_state` requirements simplifies evaluation scripts; D1 can optionally take `s₀` for better integration but still runs without it.

### 7.5 Open Questions

1. **Do we still need latent ODEs?** – D shows ≤0.30 RMSE with pure MLPs; we should benchmark against tuned C-series to decide.  
2. **Mass Constraints** – Consider a softplus-constrained head to guarantee `ṁ ≤ 0`.  
3. **Physics Iterations** – Feeding back integrated altitude/velocity into another physics pass could improve drag modeling.  

---

### 7.6 Direction D1.5: Soft-Physics Dependency Backbone

**Date**: 2025-11-25  
**Experiment**: exp8_25_11_direction_d15_soft_physics  
**Total RMSE**: **0.199** ✅ (best so far)

#### 7.6.1 What Changed Compared to Direction D1

**Key Architectural Changes:**

1. **Optional 6D Rotation Representation** (Default: Enabled)
   - **Direction D1**: Always uses 6D rotation representation with physics-aware features
   - **Direction D1.5**: Optional 6D rotation (configurable via `use_rotation_6d: true`)
   - **Purpose**: Provides smoother attitude predictions without quaternion normalization issues
   - **Implementation**: 6D rotation → rotation matrix → quaternion conversion

2. **Optional Structural Mass Monotonicity** (Default: Enabled)
   - **Direction D1**: Direct mass prediction
   - **Direction D1.5**: Optional monotonic mass via `enforce_mass_monotonicity: true`
   - **Method**: `mass_delta = -softplus(m_pred_raw)`, then `m = cumsum(mass_delta) + m0`
   - **Purpose**: Structurally enforces `m(t+1) ≤ m(t)` without requiring loss penalties

3. **Zero-Aero Handling**
   - **New Feature**: Detects zero aerodynamic coefficients and forces identity rotation
   - **Implementation**: `_is_zero_aero()` checks `Cd`, `CL_alpha`, `Cm_alpha` near zero
   - **Behavior**: When aero is zero, forces identity quaternion `[1,0,0,0]` and zero angular velocity
   - **Purpose**: Handles ballistic trajectories correctly

**Preserved Components:**
- Same shared backbone architecture (4×256 MLP)
- Same dependency chain (G3 → G2 → G1)
- Same input/output interface: `(t, context) → state`
- No initial_state required

**Loss Function Extensions:**

1. **Soft Physics Residuals**:
   - `lambda_mass_residual: 0.05` - Mass ODE residual: `dm/dt + T/(Isp*g0)`
   - `lambda_vz_residual: 0.05` - Vertical acceleration residual: `dvz/dt - a_z^phys`
   - `lambda_vxy_residual: 0.01` - Horizontal velocity residuals (light weight)

2. **Curvature Penalties**:
   - `lambda_smooth_z: 1e-4` - Second derivative penalty on altitude
   - `lambda_smooth_vz: 1e-4` - Second derivative penalty on vertical velocity

3. **Phase Schedule**:
   - Phase 1 (75% epochs): Data-only training (all soft physics weights = 0)
   - Phase 2 (25% epochs): Cosine ramp of soft physics weights to target values
   - **Purpose**: Let model learn data patterns first, then refine with physics

#### 7.6.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    └─ context: [batch, 7] ─────────────────┤
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Feature Encoding │
                                    │                  │
                                    │ 1. FourierFeatures│
                                    │    → [batch,N,17]│
                                    │                  │
                                    │ 2. ContextEncoder│
                                    │    → [batch,N,32]│
                                    │                  │
                                    │ 3. Concatenate   │
                                    │    → [batch,N,49]│
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Shared Backbone  │
                                    │ MLP 4×256        │
                                    └────────┬─────────┘
                                             │
                                             │ latent: [batch, N, 256]
                                             │
                                    ┌────────▼─────────┐
                                    │ G3: Mass Head   │
                                    │ [256→128→64→1]  │
                                    └────────┬─────────┘
                                             │
                                             │ m_pred_raw: [batch, N, 1]
                                             │
                                    ┌────────▼─────────┐
                                    │ Mass Monotonicity │
                                    │ (if enabled)     │
                                    │ -softplus(delta) │
                                    │ cumsum + m0      │
                                    └────────┬─────────┘
                                             │
                                             │ m_pred: [batch, N, 1]
                                             │
                                    ┌────────▼─────────┐
                                    │ G2: Attitude Head│
                                    │ Input: [latent||m]│
                                    │ [257→256→128→64→9]│
                                    └────────┬─────────┘
                                             │
                                             │ att_output: [batch, N, 9]
                                             │ (6D rot + 3 ω)
                                             │
                                    ┌────────▼─────────┐
                                    │ Zero-Aero Check  │
                                    │ (if zero → identity)│
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ 6D → Rotation Matrix│
                                    │ → Quaternion      │
                                    └────────┬─────────┘
                                             │
                                             │ q_pred, w_pred
                                             │
                                    ┌────────▼─────────┐
                                    │ G1: Translation  │
                                    │ Input: [latent||m||q||w]│
                                    │ [264→256→128→128→64→6]│
                                    └────────┬─────────┘
                                             │
                                             │ x,y,z,vx,vy,vz
                                             │
                                    ┌────────▼─────────┐
                                    │ Concatenate      │
                                    │ [x||v||q||w||m]  │
                                    └────────┬─────────┘
                                             │
                                             │ state: [batch, N, 14]
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Loss Function    │
                                    │ Data + Soft Phys │
                                    │ + Smoothing      │
                                    └──────────────────┘
```

#### 7.6.3 Why It Would Be Better

1. **Structural Mass Constraints**: Monotonic mass enforcement via architecture, not loss penalties
2. **Smoother Trajectories**: Curvature penalties reduce jagged altitude/velocity traces
3. **Physics Guidance**: Soft residuals guide model toward physically plausible solutions
4. **Zero-Aero Robustness**: Handles ballistic trajectories correctly
5. **Phase Training**: Data-first approach prevents physics loss from interfering with learning

#### 7.6.4 Results (exp8)

| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.199** ✅ |
| **Translation RMSE** | 0.268 |
| **Rotation RMSE** | 0.132 |
| **Mass RMSE** | 0.018 |
| **Quaternion Norm** | 1.0 (perfect) |

**Key Observations:**
- ✅ Best total RMSE achieved so far (0.199)
- ✅ Excellent mass prediction (0.018 RMSE)
- ✅ Perfect quaternion normalization
- ⚠️ Still some vertical dynamics errors (vz: 0.59, z: 0.07)

---

### 7.7 Direction D1.5.1: Position-Velocity Consistency

**Date**: 2025-11-25  
**Experiment**: exp9_25_11_direction_d151_pos_vel_consistency  
**Total RMSE**: **0.200**

#### 7.7.1 What Changed Compared to Direction D1.5

**Key Loss Function Extensions:**

1. **Position-Velocity Consistency Loss**:
   - `lambda_pos_vel: 0.5` - Enforces `v = dx/dt` relationship
   - **Purpose**: Ensures predicted velocities match position derivatives
   - **Implementation**: `L_pos_vel = ||v_pred - d(x_pred)/dt||²`

2. **Position Smoothing**:
   - `lambda_smooth_pos: 1e-3` - Second derivative penalty on all position components (x, y, z)
   - **Purpose**: Reduces position oscillations and improves trajectory smoothness
   - **Stronger than D1.5**: 10× higher weight (1e-3 vs 1e-4 for z only)

3. **Adjusted Phase Schedule**:
   - Phase 1 ratio: 60% (vs 75% in D1.5)
   - **Purpose**: Earlier introduction of physics constraints

**Preserved Components:**
- Same architecture as D1.5
- Same soft physics residuals
- Same mass monotonicity enforcement

#### 7.7.2 Why It Would Be Better

1. **Kinematic Consistency**: Position-velocity consistency ensures trajectories are kinematically valid
2. **Smoother Positions**: Position smoothing reduces oscillations in all axes
3. **Earlier Physics**: Phase schedule adjustment allows physics to guide learning earlier

#### 7.7.3 Results (exp9)

| Metric | Value |
|--------|-------|
| **Total RMSE** | 0.200 |
| **Translation RMSE** | 0.270 |
| **Rotation RMSE** | 0.131 |
| **Mass RMSE** | 0.019 |

**Key Observations:**
- Similar performance to D1.5 (0.200 vs 0.199)
- Slightly better rotation RMSE (0.131 vs 0.132)
- Position-velocity consistency helps but doesn't dramatically improve overall RMSE

---

### 7.8 Direction D1.5.2: Horizontal Motion Suppression

**Date**: 2025-11-26, 2025-11-29  
**Experiments**: exp10_26_11, exp11_29_11_direction_d152_horizontal_suppression  
**Total RMSE**: **0.198** ✅ (exp11, best overall)

#### 7.8.1 What Changed Compared to Direction D1.5.1

**Key Loss Function Extensions:**

1. **Horizontal Velocity Suppression**:
   - `lambda_zero_vxy: 1.0` - Penalizes non-zero horizontal velocities (vx, vy)
   - **Purpose**: Suppresses horizontal motion for vertical ascent trajectories
   - **Implementation**: `L_zero_vxy = ||vx||² + ||vy||²`

2. **Horizontal Acceleration Suppression**:
   - `lambda_zero_axy: 1.0` - Penalizes non-zero horizontal accelerations (ax, ay)
   - **Purpose**: Ensures horizontal forces are minimal
   - **Implementation**: `L_zero_axy = ||ax||² + ||ay||²`

3. **Horizontal Acceleration Penalty**:
   - `lambda_hacc: 0.02` - Penalizes horizontal acceleration magnitude
   - **Purpose**: Additional regularization for horizontal motion

4. **Horizontal Position Penalty**:
   - `lambda_xy_zero: 5.0` - Strong penalty on horizontal positions (x, y)
   - **Purpose**: Forces trajectories to stay near vertical axis
   - **Strongest penalty**: 5.0 weight (vs 1.0 for velocities)

5. **Enhanced Component Weights**:
   - `x: 2.0`, `y: 2.0`, `z: 3.0` - Boosted weights for position components
   - **Purpose**: Emphasize vertical motion accuracy

6. **Adjusted Phase Schedule**:
   - Phase 1 ratio: 55% (vs 60% in D1.5.1)
   - Extended training: 160 epochs (vs 120)
   - Phase 2 early stopping patience: 40 (vs 15)

**Preserved Components:**
- Same architecture as D1.5
- All D1.5.1 loss terms (position-velocity consistency, position smoothing)
- All D1.5 soft physics residuals

#### 7.8.2 Why It Would Be Better

1. **Vertical Trajectory Focus**: Horizontal suppression forces model to focus on vertical ascent
2. **Reduced Horizontal Drift**: Strong penalties prevent unwanted horizontal motion
3. **Better Vertical Accuracy**: By suppressing horizontal motion, model can allocate more capacity to vertical dynamics
4. **Extended Training**: Longer training with adjusted phase schedule allows better convergence

#### 7.8.3 Results

**exp10 (2025-11-26)**:
| Metric | Value |
|--------|-------|
| **Total RMSE** | 0.200 |
| **Translation RMSE** | 0.270 |
| **Rotation RMSE** | 0.131 |
| **Mass RMSE** | 0.019 |

**exp11 (2025-11-29)** - **Best Overall**:
| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.198** ✅✅ |
| **Translation RMSE** | 0.266 |
| **Rotation RMSE** | 0.132 |
| **Mass RMSE** | 0.015 |
| **Quaternion Norm** | 1.0 (perfect) |

**Key Observations:**
- ✅✅ Best total RMSE achieved (0.198)
- ✅ Best mass RMSE (0.015)
- ✅ Improved vertical position (z: 0.064 vs 0.074 in exp10)
- ⚠️ Slight degradation in horizontal positions (x, y) due to suppression penalties
- ✅ Perfect quaternion normalization maintained

**Position Stability Analysis (exp10 vs exp11)**:
- Z position improved by 14% (0.074 → 0.064)
- X, Y positions degraded by ~10% (due to suppression)
- Overall translation RMSE improved by 1.4%

---

### 7.9 Direction D1.5 Series Summary

**Evolution Path**: D → D1 → D1.5 → D1.5.1 → D1.5.2 → D1.5.3 → D1.5.4

| Direction | Key Innovation | Total RMSE | Best Component |
|-----------|----------------|------------|----------------|
| **D** | Dependency-aware backbone | 0.300 | Fast training |
| **D1** | Physics-aware + RK4 | 0.285 | Translation |
| **D1.5** | Soft physics + mass monotonicity | 0.199 | Mass (0.018) |
| **D1.5.1** | Position-velocity consistency | 0.200 | Rotation (0.131) |
| **D1.5.2** | Horizontal suppression | **0.198** ✅ | Mass (0.015) |
| **D1.5.3** | V2 dataloader + v2 loss | 0.198 | Mass (0.015) |
| **D1.5.4** | Central difference derivative | 0.254 | Mass (0.020) |

**Key Achievements:**
1. **Best Overall RMSE**: 0.198 (D1.5.2 exp11) - 34% better than Direction D baseline
2. **Perfect Quaternion Normalization**: All D1.5 variants maintain unit quaternions
3. **Structural Mass Constraints**: Monotonic mass without loss penalties
4. **Physics-Guided Training**: Soft residuals improve trajectory smoothness
5. **Vertical Focus**: Horizontal suppression improves vertical dynamics accuracy

**Remaining Challenges:**
1. Vertical velocity errors (vz: ~0.59) still significant
2. Rotation RMSE (0.13) could be improved further
3. Trade-off between horizontal suppression and horizontal position accuracy

Direction D offers a low-latency alternative to the hybrid stack, while Direction D1 re-introduces physics structure without Shared Stem + Latent ODE overhead. Both will act as baselines for future Direction E ideas.

---

### 7.9 Direction D1.5.3: V2 Dataloader and Loss Function

**Date**: 2025-12-01  
**Experiment**: exp14_01_12_direction_d153_position_stability  
**Total RMSE**: **0.198** ✅

#### 7.9.1 What Changed Compared to Direction D1.5.2

**Key Changes:**

1. **V2 Dataloader Integration**:
   - Uses `RocketDatasetV2` instead of `RocketDataset`
   - Loads additional features: `T_mag` (thrust magnitude) and `q_dyn` (dynamic pressure)
   - Data source: `data/processed_v2/` instead of `data/processed/`
   - **Current Status**: Models receive `T_mag` and `q_dyn` in batches but currently ignore them
   - **Future**: V2 model versions will use `InputBlockV2` to fuse v2 features

2. **V2 Loss Function Adjustments**:
   - **Reduced Soft Physics Weights**:
     - `lambda_mass_residual: 0.025` (vs 0.05 in D1.5.2)
     - `lambda_vz_residual: 0.025` (vs 0.05 in D1.5.2)
     - `lambda_vxy_residual: 0.005` (vs 0.01 in D1.5.2)
     - `lambda_smooth_z: 5.0e-5` (vs 1.0e-4 in D1.5.2)
     - `lambda_smooth_vz: 1.0e-5` (vs 1.0e-4 in D1.5.2)
   - **Position-Velocity Consistency**: `lambda_pos_vel: 0.5` (enabled)
   - **Position Smoothing**: `lambda_smooth_pos: 0.0` (disabled)
   - **Horizontal Motion Suppression**: All disabled (0.0)
     - Rationale: V2 features should naturally help with horizontal motion

**Preserved Components:**
- Same architecture as D1.5 (DirectionDPINN_D15)
- Same structural constraints (mass monotonicity, 6D rotation)
- Same phase schedule structure (55% data-only, 45% physics ramp)

#### 7.9.2 Data Flow Diagram

```
Dataset (HDF5 v2) → RocketDatasetV2
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    ├─ context: [batch, 7] ──────────────────┤
    ├─ T_mag: [batch, N] ───────────────────┤ (v2 NEW, currently unused)
    └─ q_dyn: [batch, N] ───────────────────┤ (v2 NEW, currently unused)
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Direction D1.5  │
                                    │ Architecture    │
                                    │ (same as D1.5) │
                                    └────────┬─────────┘
                                             │
                                             │ state: [batch, N, 14]
                                             │
                                    ┌────────▼─────────┐
                                    │ V2 Loss Function │
                                    │ (reduced weights)│
                                    └──────────────────┘
```

#### 7.9.3 Why It Would Be Better

1. **V2 Features Provide Physics Information**: T_mag and q_dyn encode physics-critical information directly, reducing information bottleneck
2. **Lighter Physics Losses**: Since v2 features provide physics info, explicit physics loss penalties can be lighter
3. **Simplified Loss Function**: Disabled horizontal suppression and position smoothing (v2 features should naturally help)
4. **Future-Proof**: Prepares for v2 model versions that will actually use T_mag and q_dyn via `InputBlockV2`

#### 7.9.4 Results (exp14)

| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.198** ✅ |
| **Translation RMSE** | 0.266 |
| **Rotation RMSE** | 0.132 |
| **Mass RMSE** | 0.015 |
| **Quaternion Norm** | 1.0 (perfect) |

**Key Observations:**
- ✅ Matches D1.5.2 performance (0.198 RMSE)
- ✅ Excellent mass prediction (0.015 RMSE)
- ✅ Perfect quaternion normalization
- ⚠️ V2 features currently unused (future: v2 model versions will use them)

---

### 7.10 Direction D1.5.4: Central Difference Derivative Method

**Date**: 2025-12-05  
**Experiment**: exp17_05_12_direction_d154_central_diff_scaled  
**Total RMSE**: **0.254**

#### 7.10.1 What Changed Compared to Direction D1.5.3

**Key Changes:**

1. **Derivative Computation Method**:
   - **Previous (D1.5.3)**: Forward difference `ds/dt = (s(t+1) - s(t)) / dt`
   - **D1.5.4**: Central difference `ds/dt = (s(t+1) - s(t-1)) / (2*dt)`
   - **Rationale**: Central difference provides smoother and more accurate derivatives (O(h²) vs O(h) truncation error)

2. **New Loss Function Class**:
   - Uses `PINNLossV2` instead of `PINNLoss`
   - Overrides `compute_derivative()` method to use `central_difference()`
   - Handles non-uniform time grids correctly

3. **Component Scaling for Physics Residuals**:
   - `physics_scale`: Scales physics residuals by component type
     - Default: `pos: 1.0, vel: 1.0, quat: 0.1, ang: 0.2, mass: 1e-3`
   - **Purpose**: Balances physics loss contributions across different state components

4. **Reweighted Physics Terms**:
   - `physics_groups`: Weights for physics loss groups
     - Default: `pos: 1.0, vel: 1.0, quat: 0.2, ang: 0.5, mass: 1.0`
   - **Purpose**: Fine-tune physics loss weighting for better convergence

**Preserved Components:**
- Same architecture as D1.5.3 (DirectionDPINN_D15)
- Same structural constraints (mass monotonicity, 6D rotation)
- Same phase schedule structure (55% data-only, 45% physics ramp)
- All D1.5.3 loss parameters preserved

#### 7.10.2 Derivative Method Comparison

**Forward Difference (v1, D1.5.3)**:
```python
# For interior points
ds/dt[i] = (s[i+1] - s[i]) / (t[i+1] - t[i])
```
- **Pros**: Simple, only needs next point
- **Cons**: Less accurate (O(h) truncation error), asymmetric (only uses future information)

**Central Difference (v2, D1.5.4)**:
```python
# For interior points
ds/dt[i] = (s[i+1] - s[i-1]) / (2 * (t[i+1] - t[i-1]))
```
- **Pros**: More accurate (O(h²) truncation error), symmetric (uses both past and future)
- **Cons**: Requires both previous and next points (not applicable at boundaries)

**Boundary Handling**:
- **First point**: Forward difference `(s[1] - s[0]) / (t[1] - t[0])`
- **Last point**: Backward difference `(s[N-1] - s[N-2]) / (t[N-1] - t[N-2])`
- **Interior points**: Central difference

#### 7.10.3 Why It Would Be Better

1. **More Accurate Derivatives**: Central difference has O(h²) truncation error vs O(h) for forward difference
2. **Smoother Gradients**: Symmetric method reduces numerical noise in derivative computation
3. **Better Physics Residuals**: More accurate derivatives lead to more accurate physics residual computation
4. **Component Balance**: Scaling and reweighting help balance physics loss contributions
5. **Non-Uniform Grids**: Central difference implementation handles variable time steps correctly

#### 7.10.4 Results (exp17)

| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.254** |
| **Translation RMSE** | 0.330 |
| **Rotation RMSE** | 0.189 |
| **Mass RMSE** | 0.020 |
| **Quaternion Norm** | 1.0 (perfect) |

**Key Observations:**
- ⚠️ Higher RMSE than D1.5.3 (0.254 vs 0.198)
- ✅ Perfect quaternion normalization maintained
- ⚠️ Rotation RMSE increased (0.189 vs 0.132 in D1.5.3)
- ⚠️ Translation RMSE increased (0.330 vs 0.266 in D1.5.3)
- **Note**: This may indicate that central difference requires different loss weight tuning, or that the current configuration needs adjustment

**Potential Reasons for Higher RMSE:**
1. **Hyperparameter Sensitivity**: Central difference may require different loss weights
2. **Physics Scale Tuning**: Default physics_scale values may not be optimal
3. **Training Dynamics**: More accurate derivatives may change training dynamics, requiring different schedules
4. **Component Balance**: Physics groups may need rebalancing for central difference

---

## 8. Direction AN: Shared Stem + Mission Branches + Physics Residuals

**Date**: 2025-12-02  
**Experiments**: exp15_02_12_direction_an_baseline, exp16_04_12_direction_an_v2, exp17_04_12_direction_an_v2  
**Total RMSE**: **0.197** ✅ (exp15, exp16, exp17)

### 8.1 What Changed Compared to Direction D1.5

**Key Architectural Changes:**

1. **ANSharedStem** (Replaces Direction D's backbone):
   - **Direction D1.5**: Shared backbone MLP (4×256) with dependency heads
   - **Direction AN**: ANSharedStem with residual MLP stack
     - Fourier time features (8 frequencies → 17D)
     - ContextEncoder (7 → 128, Tanh)
     - Residual MLP stack (4 layers × 128, Tanh, LayerNorm)
     - **Purpose**: Unified feature extraction with residual connections for stability

2. **Mission Branches** (Independent, not dependency-ordered):
   - **Direction D1.5**: Dependency-ordered heads (G3 → G2 → G1)
   - **Direction AN**: Independent branches (all receive same latent)
     - TranslationBranch: `[128→128→128→6]` for `[x, y, z, vx, vy, vz]`
     - RotationBranch: `[128→256→256→7]` for `[q0, q1, q2, q3, wx, wy, wz]`
     - MassBranch: `[128→64→1]` for `[m]`
   - **Key Difference**: No dependency chain; all branches operate independently on shared latent

3. **Physics Residual Layer** (New Component):
   - **Direction D1.5**: Soft physics residuals computed in loss function
   - **Direction AN**: Explicit `PhysicsResidualLayer` that computes residuals using autograd
     - Uses `compute_dynamics` from physics library
     - Returns `PhysicsResiduals` dataclass with ODE residuals
     - **Purpose**: Provides physics residuals directly from forward pass, not just in loss

4. **Dual Output**:
   - **Direction D1.5**: Returns only state predictions
   - **Direction AN**: Returns both `(state_pred, physics_residuals)`
     - Enables flexible loss computation (can weight residuals differently)
     - Physics residuals available for analysis and debugging

**Preserved Components:**
- Same input/output interface: `(t, context) → state`
- Quaternion normalization (applied after rotation branch)
- No initial_state required

### 8.2 Data Flow Diagram

```
Dataset (HDF5) → DataLoader
    │
    ├─ t: [batch, N, 1] ────────────────────┐
    ├─ context: [batch, 7] ─────────────────┤
    └─ control: [batch, N, 4] (optional) ─┤
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ ANSharedStem     │
                                    │                  │
                                    │ 1. TimeEmbedding │
                                    │    (Fourier)     │
                                    │    → [batch,N,17]│
                                    │                  │
                                    │ 2. ContextEncoder│
                                    │    → [batch,N,128]│
                                    │                  │
                                    │ 3. Concatenate   │
                                    │    → [batch,N,145]│
                                    │                  │
                                    │ 4. Residual MLP  │
                                    │    4 layers×128  │
                                    │    Tanh, LayerNorm│
                                    │    → [batch,N,128]│
                                    └────────┬─────────┘
                                             │
                                             │ latent: [batch, N, 128]
                                             │
                                    ┌────────▼─────────┐
                                    │ Mission Branches │
                                    │                  │
                                    │ ┌──────┬──────┬─┐│
                                    │ │Trans │Rot   │M││
                                    │ │Branch│Branch│ ││
                                    │ │[128→ │[128→ │[││
                                    │ │128→  │256→  │1││
                                    │ │128→6]│256→7]│2││
                                    │ │      │      │8││
                                    │ │      │      │→││
                                    │ │      │      │1││
                                    │ └──┬───┴──┬───┴─┘│
                                    │    │      │      │
                                    │ trans:│rotation:│mass:│
                                    │[batch,│[batch,  │[batch,│
                                    │N,6]   │N,7]     │N,1]  │
                                    └───────┴─────────┴──────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Quaternion Norm  │
                                    │ normalize_quat()│
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Pack State        │
                                    │ [trans||rot||mass]│
                                    └────────┬─────────┘
                                             │
                                             │ state: [batch, N, 14]
                                             │
                                    ┌────────▼─────────┐
                                    │ Physics Residual │
                                    │ Layer            │
                                    │ (autograd-based) │
                                    │ compute_dynamics │
                                    └────────┬─────────┘
                                             │
                                             │ physics_residuals
                                             │
                                    ┌────────▼─────────┐
                                    │ Output           │
                                    │ (state, residuals)│
                                    └──────────────────┘
```

### 8.3 Why It Would Be Better

1. **Residual Connections**: ANSharedStem uses residual MLP stack for better gradient flow and training stability
2. **Independent Branches**: Mission branches operate independently, avoiding dependency chain complexity
3. **Explicit Physics Residuals**: Physics residuals computed in forward pass, enabling flexible loss weighting
4. **Unified Stem**: Single shared stem processes all features together, similar to C2 but simpler (no Transformer)
5. **Physics Integration**: Physics residual layer uses actual physics library, ensuring consistency with WP1

### 8.4 Results

**exp15 (2025-12-02) - Baseline**:
| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.197** ✅ |
| **Translation RMSE** | 0.264 |
| **Rotation RMSE** | 0.133 |
| **Mass RMSE** | 0.015 |
| **Quaternion Norm** | 1.0 (perfect) |

**exp16 (2025-12-04) - V2**:
| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.197** ✅ |
| **Translation RMSE** | 0.264 |
| **Rotation RMSE** | 0.133 |
| **Mass RMSE** | 0.015 |
| **Quaternion Norm** | 1.0 (perfect) |

**exp17 (2025-12-04) - V2 (continued)**:
| Metric | Value |
|--------|-------|
| **Total RMSE** | **0.197** ✅ |
| **Translation RMSE** | 0.264 |
| **Rotation RMSE** | 0.133 |
| **Mass RMSE** | 0.015 |
| **Quaternion Norm** | 1.0 (perfect) |

**Key Observations:**
- ✅ Excellent performance (0.197 RMSE) - matches D1.5.2/D1.5.3
- ✅ Best mass RMSE (0.015) - tied with D1.5.2
- ✅ Perfect quaternion normalization
- ✅ Consistent performance across v1 and v2 dataloaders
- ✅ Physics residuals available for analysis

### 8.5 Potential Drawbacks

1. **No Dependency Chain**: Independent branches may miss physics dependencies (mass → attitude → translation)
2. **Simpler Stem**: No Transformer or attention mechanism (vs C2's Shared Stem)
3. **Physics Residual Overhead**: Computing residuals in forward pass adds computational cost
4. **Less Specialized**: Branches don't receive specialized inputs (vs D1.5's dependency chain)

### 8.6 V2 Support

Direction AN supports v2 dataloader:
- **AN (v1)**: Uses `ANSharedStem` with time + context only
- **AN1 (v2)**: Uses `InputBlockV2` to fuse T_mag and q_dyn (future implementation)
- **Current Status**: AN accepts but ignores v2 features (same as D1.5.3)

---
