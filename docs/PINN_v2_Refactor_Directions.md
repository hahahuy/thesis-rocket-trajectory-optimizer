````markdown
# PINN v2 Refactor Plan  
**File:** `docs/PINN_v2_Refactor_Directions.md`  
**Scope:** 6-DOF Rocket Trajectory PINN – architecture redesign to improve RMSE and stability.

---

## 0. Purpose & Big Picture

Our current PINN:

- Input: time `t` (Fourier features) + context vector
- Core: plain MLP (6 × 128, tanh)
- Output: full 14D state per time step
- Loss: data MSE + physics residual + boundary MSE

**Observed issue:** RMSE ≈ 0.9 (≈ 90% off scale of normalized states), instability especially in orientation and coupled dynamics → architecture does not exploit temporal structure and physics enough.

This document proposes **three refactor directions**, all compatible with the current codebase:

1. **Direction A – Physics-based:** Latent Neural ODE PINN  
2. **Direction B – Data-based:** Sequence model (Transformer / TCN)  
3. **Direction C – Hybrid:** Sequence + Latent ODE + PINN

For each direction:

- What changes conceptually  
- Where to change the code  
- How to hook into training and losses  
- What to log, how to debug, and how to note function changes

---

## 1. Current Baseline (Quick Recap)

### 1.1 Code Map

- **Model:**
  - `src/models/pinn.py`
  - `src/models/architectures.py`  
    - `FourierFeatures`
    - `ContextEncoder`
    - `MLP`, `MLPBlock`
- **Physics:**
  - `src/physics/dynamics_pytorch.py`  
    - `compute_dynamics(state, control, params, scales)`
- **Losses:**
  - `src/train/losses.py`
    - `PINNLoss.data_loss`
    - `PINNLoss.physics_loss`
    - `PINNLoss.boundary_loss`
- **Training:**
  - `src/train/train_pinn.py`
    - `train_epoch(...)`
    - `train(...)` or main loop (depending on file)
- **Data:**
  - `src/utils/loaders.py`
    - `RocketDataset`
- **Eval & Plots:**
  - `src/eval/visualize_pinn.py`

### 1.2 Current Model Flow

```text
(t, context)  ──► time_embedding(FourierFeatures)
       │
       └─────► context_encoder(ContextEncoder)
                     │
       [time_emb || ctx_emb] ──► MLP ──► s_pred ∈ ℝ¹⁴
````

Loss:

```text
L = λ_data * L_data + λ_phys * L_phys + λ_bc * L_bc
```

---

## 2. Shared Refactor Guidelines (For All Directions)

### 2.1 Where to Document Changes

**File to create (if not already):**

* `docs/ARCHITECTURE_CHANGELOG.md`

For every non-trivial change:

* Add an entry:

```markdown
## [YYYY-MM-DD] <your_name>
- Modified: src/models/pinn.py
  - Changed forward() to output (state, optional_latent).
  - Reason: prepare for latent ODE backbone (Direction A).
- Added: src/models/latent_ode.py
  - New class: RocketLatentODEPINN
```

**Inside code**, always:

* Add a **header comment** above modified functions:

```python
# [PINN_V2][YYYY-MM-DD][your_name]
# Change: describe briefly the change and the direction (A/B/C)
```

### 2.2 Branching Strategy

* Use branches:

  * `feature/pinn_v2_direction_A`
  * `feature/pinn_v2_direction_B`
  * `feature/pinn_v2_direction_C`
* Do **small PRs**:

  * First PR: architecture only + unit tests
  * Second PR: hyperparameter tuning
  * Third PR: integration with visualization & metrics

### 2.3 Debugging Checklist (Core)

Always log and check:

* Per-component RMSE (already implemented in `visualize_pinn.py`)
* Time-series plots for:

  * altitude (z)
  * velocity norm
  * quaternion norm `||q||`
  * mass
* Physics residual histogram: distribution of `ds/dt_num - f(s)`
  *(optional extra log – add later)*

When something breaks:

1. Check **normalization scales** used in preprocessing vs. model vs. eval.
2. Check `compute_dynamics` by running it on a **frozen ground-truth trajectory** and verifying the residual is small.
3. Log sample trajectories before and after each big refactor.

---

## 3. Direction A – Physics-Based Latent Neural ODE PINN

### 3.1 Concept

Instead of predicting `s(t)` directly from `(t, context)`, we:

1. Encode context into initial latent state `z0`.
2. Evolve `z(t)` through a **neural ODE**:
   [
   \dot{z}(t) = g_\theta(z(t), t, c)
   ]
3. Decode latent `z(t)` back to physical state `s(t)`.

The PINN losses are applied on decoded states `s(t)` as before.

This aligns architecture with the fact that trajectories follow ODEs.

### 3.2 High-Level Architecture

```text
(context) ──► ContextEncoder_z ──► z0 (latent initial state)

t_grid ─────────────────────────► ODE solver:
                                  dz/dt = gθ(z, t, ctx_emb)
                                  → z(t_0..t_N)

z(t_i) ──► Decoder ──► s_pred(t_i)
```

### 3.3 Where & How to Change the Code

#### 3.3.1 New File: `src/models/latent_ode.py`

Add:

* `class LatentDynamicsNet(nn.Module)`

  * Input: latent `z` + maybe time embedding + context embedding
  * Output: `dz/dt`
* `class LatentODEBlock(nn.Module)`

  * Wraps a time integrator like fixed-step Euler or RK4 (we can start with fixed-step Euler using the time grid from dataset).
* `class RocketLatentODEPINN(nn.Module)`

  * Contains:

    * `FourierFeatures` (reused) or simple `t` usage
    * `ContextEncoder` → `ctx_emb`
    * `ContextEncoder_z` → `z0` (another small MLP)
    * `LatentDynamicsNet`
    * `Decoder` (MLP to state space)

Pseudo-structure:

```python
class RocketLatentODEPINN(nn.Module):
    def __init__(self, latent_dim=64, ...):
        super().__init__()
        self.time_embedding = FourierFeatures(n_frequencies=8)
        self.context_encoder = ContextEncoder(context_dim, embedding_dim=64)
        self.z0_encoder = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.Tanh()
        )
        self.dynamics_net = LatentDynamicsNet(latent_dim, 64 + 17)  # latent + (ctx_emb + t_emb)
        self.decoder = MLP(input_dim=latent_dim, output_dim=14, ...)
    
    def forward(self, t, context):
        # context embedding
        ctx_emb = self.context_encoder(context)     # [..., 64]

        # z0 from context
        z0 = self.z0_encoder(ctx_emb)               # [..., latent_dim]

        # Broadcast z0 over time steps
        # t: [batch, N, 1] → t_emb: [batch, N, time_dim]
        t_emb = self.time_embedding(t)

        # Run custom ODE integration using z0, t_emb, ctx_emb
        # z: [batch, N, latent_dim]
        z_traj = integrate_latent_ode(
            z0, t, t_emb, ctx_emb, self.dynamics_net
        )

        # Decode
        state = self.decoder(z_traj)                # [batch, N, 14]
        return state
```

**Note:** start with **simple fixed-step Euler** integration using the known time grid; avoid adding `torchdiffeq` initially to control complexity.

#### 3.3.2 Modify Training Driver: `src/train/train_pinn.py`

Add configuration flag:

* In argument parsing or config: `--model_type {mlp, latent_ode}`

Then, where the model is instantiated:

```python
if cfg.model_type == "mlp":
    model = PINNModel(...)  # existing
elif cfg.model_type == "latent_ode":
    from src.models.latent_ode import RocketLatentODEPINN
    model = RocketLatentODEPINN(
        context_dim=..., latent_dim=64, ...
    )
```

No changes are needed in `PINNLoss` as long as `forward()` still returns state of shape `[batch, N, 14]`.

#### 3.3.3 Logging & Debug Instructions (Direction A)

Add debug plots for:

* Norm of `z(t)` over time (detect exploding latent dynamics)
* Compare:

  * baseline MLP vs. latent ODE trajectories for same context

Log in `docs/ARCHITECTURE_CHANGELOG.md`:

```markdown
## [YYYY-MM-DD] Direction A – Latent ODE
- Added RocketLatentODEPINN to src/models/latent_ode.py
- Updated train_pinn.py to accept `model_type=latent_ode`.
- TODO: benchmark RMSE on test set and compare with MLP baseline.
```

---

## 4. Direction B – Data-Based Sequence Model (Transformer / TCN)

### 4.1 Concept

Instead of predicting each time step independently, we treat the full time grid as a **sequence** and use a **Transformer or Temporal ConvNet** to model temporal dependencies.

Model:

```text
(t_i, context) ──► embeddings ──► sequence model (Transformer/TCN) ──► state sequence
```

This makes the model aware that `s(t_i)` and `s(t_{i+1})` are highly correlated.

### 4.2 High-Level Architecture

```text
Time embedding: t_emb[i] = FourierFeatures(t_i)      (shape: [N, time_dim])
Context embedding: ctx_emb = ContextEncoder(context) (shape: [ctx_dim])
Broadcast: ctx_seq[i] = ctx_emb                      (shape: [N, ctx_dim])

Input sequence: seq[i] = [t_emb[i] || ctx_seq[i]]    (shape: [N, d_in])

Sequence Model:
    - Option 1: TransformerEncoder
    - Option 2: TCN (1D ConvNet over time)

Output: seq_out[i] → MLP head → s_pred(t_i)
```

### 4.3 Where & How to Change the Code

#### 4.3.1 New File: `src/models/sequence_pinn.py`

Add:

* `class RocketSequencePINN(nn.Module)`

Structure:

```python
class RocketSequencePINN(nn.Module):
    def __init__(self, context_dim, d_model=128, n_layers=4, n_heads=4, ...):
        super().__init__()
        self.time_embedding = FourierFeatures(n_frequencies=8)
        self.context_encoder = ContextEncoder(context_dim, embedding_dim=64)
        
        self.input_proj = nn.Linear(17 + 64, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.output_head = MLP(
            input_dim=d_model,
            output_dim=14,
            hidden_dims=[128, 128],
            activation="tanh"
        )
    
    def forward(self, t, context):
        # t: [batch, N, 1]
        # context: [batch, context_dim]

        t_emb = self.time_embedding(t)                 # [batch, N, 17]
        
        if context.dim() == 2 and t.dim() == 3:
            b, N, _ = t.shape
            ctx = context.unsqueeze(1).expand(b, N, -1)  # [batch, N, context_dim]
        else:
            ctx = context

        ctx_emb = self.context_encoder(ctx)           # [batch, N, 64]

        x = torch.cat([t_emb, ctx_emb], dim=-1)       # [batch, N, 17+64]
        x = self.input_proj(x)                        # [batch, N, d_model]

        x = self.transformer(x)                       # [batch, N, d_model]
        state = self.output_head(x)                   # [batch, N, 14]
        return state
```

> If GPU memory is an issue, we can switch Transformer → TCN later.

#### 4.3.2 Hook into Training

In `src/train/train_pinn.py`, add:

```python
elif cfg.model_type == "sequence":
    from src.models.sequence_pinn import RocketSequencePINN
    model = RocketSequencePINN(context_dim=..., ...)
```

No changes to `PINNLoss`.

#### 4.3.3 Debugging & Logging (Direction B)

Additional logs:

* Attention maps (if using Transformer):

  * For one batch, log attention weights for early vs. late layers.
* Compare:

  * Baseline MLP vs. Sequence model:

    * RMSE per component
    * Visual smoothness of trajectory

In `ARCHITECTURE_CHANGELOG.md`:

```markdown
## [YYYY-MM-DD] Direction B – Sequence model
- Added RocketSequencePINN using TransformerEncoder.
- Updated train_pinn.py (`model_type=sequence`).
- Next: add scripts to visualize attention maps on selected test cases.
```

---

## 5. Direction C – Hybrid: Sequence + Latent ODE + PINN

### 5.1 Concept

Combine strengths of A & B:

* Sequence model (Transformer/TCN) to capture high-level temporal patterns.
* Latent ODE modeling the underlying continuous dynamics.
* PINN loss to enforce physics.

Two viable hybrid variants:

1. **Sequence → Latent ODE**

   * Sequence model processes context and initial short segment → produce good initial latent `z0`.
2. **Latent ODE → Sequence Residual**

   * Latent ODE provides physics-consistent baseline; sequence model corrects residual errors.

Here we describe Variant 1 (simpler to implement).

### 5.2 High-Level Architecture (Variant 1)

```text
context ──► ContextEncoder → ctx_emb
t_short, s_short_true ──► EncoderSeq → z0 (latent initial state)

Full time grid t_full:
    integrate z(t) from z0 using LatentDynamicsNet
    z_full(t_i) → Decoder → s_pred(t_i)

Loss: PINN (data + physics + boundary) on s_pred.
```

This is like Direction A, but with a **learned encoder for z0** that uses the first part of the trajectory or richer context.

### 5.3 Where & How to Change the Code

#### 5.3.1 Extend `RocketLatentODEPINN` → `RocketHybridPINN`

Create new class in `src/models/latent_ode.py` or `src/models/hybrid_pinn.py`:

```python
class RocketHybridPINN(nn.Module):
    def __init__(self, context_dim, latent_dim=64, ...):
        super().__init__()
        self.context_encoder = ContextEncoder(context_dim, embedding_dim=64)
        self.time_embedding = FourierFeatures(n_frequencies=8)

        self.encoder_seq = TransformerEncoderForZ0(
            input_dim=17 + 64,
            d_model=128,
            ...
        )

        self.z0_proj = nn.Linear(128, latent_dim)
        self.dynamics_net = LatentDynamicsNet(latent_dim, ...)
        self.decoder = MLP(input_dim=latent_dim, output_dim=14, ...)
    
    def forward(self, t, context, state_short=None):
        """
        t: [batch, N, 1]
        context: [batch, context_dim]
        state_short: optional, [batch, N0, 14] for encoder use (future extension)
        """

        ctx_emb = self.context_encoder(context)  # [batch, 64]
        b, N, _ = t.shape
        ctx_seq = ctx_emb.unsqueeze(1).expand(b, N, -1)

        t_emb = self.time_embedding(t)          # [batch, N, 17]
        enc_input = torch.cat([t_emb, ctx_seq], dim=-1)

        # Option 1: use first N0 time steps as sequence to compute z0
        N0 = min(10, N)  # or config parameter
        z0_seq_features = self.encoder_seq(enc_input[:, :N0, :])  # [batch, N0, d_model]
        z0_pooled = z0_seq_features.mean(dim=1)                    # [batch, d_model]
        z0 = self.z0_proj(z0_pooled)                               # [batch, latent_dim]

        # Now integrate Latent ODE over full time grid
        z_traj = integrate_latent_ode(z0, t, t_emb, ctx_emb, self.dynamics_net)
        state = self.decoder(z_traj)
        return state
```

You can start **without using `state_short`** and only use context & time info for simplicity.

#### 5.3.2 Training Integration

In `train_pinn.py`:

```python
elif cfg.model_type == "hybrid":
    from src.models.hybrid_pinn import RocketHybridPINN
    model = RocketHybridPINN(context_dim=..., ...)
```

**Important:**
Hybrid model is more complex; start with **small hyperparameters** (lower `latent_dim`, fewer Transformer layers).

#### 5.3.3 Debugging & Logging (Direction C)

* Log:

  * latent initial state `z0` statistics (mean, std per dimension)
  * Norm of z(t) vs t
  * Compare:

    * MLP vs Latent ODE vs Hybrid (3 curves on same plot)

In `ARCHITECTURE_CHANGELOG.md`:

```markdown
## [YYYY-MM-DD] Direction C – Hybrid PINN
- Added RocketHybridPINN combining Transformer encoder for z0 and Latent ODE dynamics.
- Integrated with train_pinn via `model_type=hybrid`.
- TODO: ablation study:
  - hybrid vs pure latent_ode vs pure sequence
  - impact of N0 (number of timesteps used in z0 encoder)
```

---

## 6. Shared Output & Evaluation Changes

Regardless of direction:

1. **RMSE Breakdown:**
   Ensure `evaluate_model()` continues to compute RMSE per component. Log these to a file like:

   * `experiments/.../metrics_direction_<A|B|C>.json`

2. **Config & Naming:**
   When running experiments, name them clearly:

   * `expA_latent_ode_<date>`
   * `expB_transformer_<date>`
   * `expC_hybrid_<date>`

3. **Debug Plots:**

   Extend `src/eval/visualize_pinn.py` to optionally:

   * plot quaternion norm
   * plot physics residual (optional extra function)

---

## 7. Practical Roadmap

To avoid chaos, implement directions **in this order**:

1. **Step 1 – Safer Output & Context**

   * Add:

     * Better `ContextEncoder` (2–3 layers)
     * Quaternion normalization in output
     * Split heads (translation, rotation, mass)
   * This can be done inside current `PINN` class.

2. **Step 2 – Direction A or B**

   * Choose **one** direction (A: Latent ODE or B: Sequence).
   * Implement, document, compare.

3. **Step 3 – Direction C (Hybrid)**

   * Only if A or B gives good baseline and we want maximum performance.

---

## 8. Summary for Workers

* **You do NOT need to touch the data loader or loss functions** to try new architectures, as long as the model still outputs `[batch, N, 14]`.

* The **main modifications** live in:

  * `src/models/pinn.py` (baseline MLP variants)
  * `src/models/latent_ode.py` (Direction A & C)
  * `src/models/sequence_pinn.py` (Direction B)
  * `src/models/hybrid_pinn.py` (Direction C, optional)

* **Always:**

  * Note changes in `docs/ARCHITECTURE_CHANGELOG.md`
  * Add comments with tags: `# [PINN_V2][YYYY-MM-DD][name]`
  * Run evaluation and save metrics JSON for every new model type
  * Produce at least:

    * 3 trajectory plots (different cases)
    * RMSE report
    * Short comment in the changelog: “Better / worse than baseline, why we think so.”

This MD file should serve as the **central reference** for implementing and debugging the three main architecture directions for PINN v2.

---

## 9. How to Test All 3 Directions (Simple and Mandatory)

This section tells workers **exactly how to run experiments** and compare results across A/B/C.

### 9.1 Create 3 Config Files

Create:
configs/model_A_latent_ode.yaml
configs/model_B_sequence.yaml
configs/model_C_hybrid.yaml

Each config must define:

```yaml
model_type: latent_ode   # or sequence / hybrid
```
All other settings (dataset, loss weights, optimizer) remain identical.

### 9.2 Train Each Model Separately
Run: 
```bash
python train.py --config configs/model_A_latent_ode.yaml
python train.py --config configs/model_B_sequence.yaml
python train.py --config configs/model_C_hybrid.yaml
```
Each run will produce:

```bash 
experiments/expA_latent_ode/
experiments/expB_sequence/
experiments/expC_hybrid/
```
Inside each folder you will find:

+ checkpoints/best.pt

+ logs/train_log.json

+ figures/* (optional)
Use the same evaluation script for all

```
```
