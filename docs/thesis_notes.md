# Development Stage of the PINN model

## Table of Contents
1. [Data](#data)
2. [Model Structure](#model-structure)
   - 1. [Base Model](#1-base-model)
   - 2. [Direction A: Latent Neural ODE PINN](#2-direction-a-latent-neural-ode-pinn)
   - 3. [Direction B: Sequence Model (Transformer) PINN](#3-direction-b-sequence-model-transformer-pinn)
   - 4. [Direction C: Hybrid PINN](#4-direction-c-hybrid-pinn)
   - 5. [Direction C1: Enhanced Hybrid PINN](#5-direction-c1-enhanced-hybrid-pinn)

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