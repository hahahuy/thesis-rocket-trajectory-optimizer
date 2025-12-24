# PINN Architecture Diagrams

This document contains Mermaid diagrams for all PINN architecture variants, from the baseline model through Directions A, B, C, C1, C2, C3, and the dependency-aware Directions D/D1.

## Baseline PINN

The original vanilla MLP architecture that serves as the reference baseline.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures<br/>8 frequencies"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→16, Tanh"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×16"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×16"]
    F --> G
    G --> H["Concatenate<br/>t_emb || ctx_emb"]
    H --> I["MLP<br/>6 layers × 128 neurons<br/>tanh, LayerNorm, Dropout"]
    I --> J["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style J fill:#c8e6c9,color:#000000
```

## Direction A: Latent Neural ODE PINN

Encodes context into initial latent state, evolves through neural ODE, then decodes to physical state.

```mermaid
graph TD
    A["context: batch×7"] --> B["ContextEncoder<br/>7→64"]
    B --> C["z0 Encoder<br/>64→64, Tanh"]
    C --> D["z0: batch×64"]
    
    E["t: batch×N×1"] --> F["TimeEmbedding<br/>FourierFeatures"]
    F --> G["t_emb: batch×N×17"]
    B --> H["ctx_emb: batch×64"]
    
    D --> I["Latent ODE Block<br/>Euler Integration"]
    G --> I
    H --> I
    I --> J["z_traj: batch×N×64"]
    
    J --> K["Decoder<br/>3 layers × 128 neurons"]
    K --> L["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style E fill:#e1f5ff,color:#000000
    style D fill:#fff9c4,color:#000000
    style J fill:#fff9c4,color:#000000
    style L fill:#c8e6c9,color:#000000
```

## Direction B: Sequence Transformer PINN

Treats the full time grid as a sequence and models temporal dependencies via Transformer encoder.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures<br/>8 frequencies"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→64, Tanh"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×64"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×64"]
    F --> G
    G --> H["Concatenate<br/>t_emb || ctx_emb"]
    H --> I["Input Projection<br/>81→128"]
    I --> J["Transformer Encoder<br/>4 layers, 4 heads<br/>d_model=128, GELU"]
    J --> K["Output Head<br/>MLP: 128→128→14"]
    K --> L["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style J fill:#ffccbc,color:#000000
    style L fill:#c8e6c9,color:#000000
```

## Direction C: Hybrid Sequence + Latent ODE PINN

Combines Transformer encoder for z0 estimation with Latent ODE dynamics.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→64"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×64"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×64"]
    F --> G
    G --> H["Concatenate<br/>t_emb || ctx_emb"]
    H --> I["Transformer Encoder<br/>First N0 timesteps<br/>for z0 estimation"]
    I --> J["z0 Pooling<br/>Mean over N0"]
    J --> K["z0 Projection<br/>128→64"]
    K --> L["z0: batch×64"]
    
    E --> M["Full Time Grid"]
    F --> M
    L --> N["Latent ODE Block<br/>Euler Integration"]
    M --> N
    N --> O["z_traj: batch×N×64"]
    
    O --> P["Decoder<br/>3 layers × 128 neurons"]
    P --> Q["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style I fill:#ffccbc,color:#000000
    style L fill:#fff9c4,color:#000000
    style O fill:#fff9c4,color:#000000
    style Q fill:#c8e6c9,color:#000000
```

## Direction C1: Enhanced Hybrid PINN

Adds Set#1 (Output Stability) and Set#2 (Deep Context Encoder) improvements.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures"]
    C["context: batch×7"] --> D["DeepContextEncoder<br/>Set#2: 7→64→128→128→64→32<br/>GELU, LayerNorm"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×32"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×32"]
    F --> G
    G --> H["Concatenate<br/>t_emb || ctx_emb"]
    H --> I["Transformer Encoder<br/>First N0 timesteps<br/>for z0 estimation"]
    I --> J["z0 Pooling<br/>Mean over N0"]
    J --> K["z0 Projection<br/>128→64"]
    K --> L["z0: batch×64"]
    
    E --> M["Full Time Grid"]
    F --> M
    L --> N["Latent ODE Block<br/>Euler Integration"]
    M --> N
    N --> O["z_traj: batch×N×64"]
    
    O --> P["Output Heads<br/>Set#1: Split Heads"]
    P --> Q1["Translation Head<br/>64→128→128→6"]
    P --> Q2["Rotation Head<br/>64→128→128→7"]
    P --> Q3["Mass Head<br/>64→64→1"]
    
    Q1 --> R1["Δ_translation: 6D"]
    Q2 --> R2["Δ_rotation: 7D"]
    Q3 --> R3["Δ_mass: 1D"]
    
    S["s0: batch×14"] --> T["Δ-State Reconstruction<br/>Set#1: s = s0 + Δs"]
    R1 --> T
    R2 --> T
    R3 --> T
    T --> U["Quaternion Normalization<br/>Set#1"]
    U --> V["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style S fill:#e1f5ff,color:#000000
    style D fill:#ffccbc,color:#000000
    style I fill:#ffccbc,color:#000000
    style L fill:#fff9c4,color:#000000
    style O fill:#fff9c4,color:#000000
    style P fill:#f8bbd0,color:#000000
    style V fill:#c8e6c9,color:#000000
```

## Direction C2: Shared Stem + Dedicated Branches

Introduces Shared Stem for unified temporal+context processing and Dedicated Branches for specialized outputs.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures<br/>8 frequencies"]
    C["context: batch×7"] --> D["DeepContextEncoder<br/>Set#2: 7→64→128→128→64→32"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×32"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×32"]
    F --> G
    G --> H["Input Projection<br/>49→128"]
    H --> I["Shared Stem<br/>Transformer Encoder<br/>4 layers, 4 heads<br/>d_model=128"]
    I --> J["shared_emb: batch×N×128"]
    
    J --> K["z0 Derivation<br/>First N0 timesteps<br/>Mean pooling"]
    K --> L["z0 Projection<br/>128→64"]
    L --> M["z0: batch×64"]
    
    J --> N["Full Sequence"]
    M --> O["Latent ODE Block<br/>Euler Integration"]
    N --> O
    O --> P["z_traj: batch×N×64"]
    
    P --> Q1["TranslationBranch<br/>64→128→128→6"]
    P --> Q2["RotationBranch<br/>64→256→256→7"]
    P --> Q3["MassBranch<br/>64→64→1"]
    
    Q1 --> R1["Δ_translation: 6D"]
    Q2 --> R2["Δ_rotation: 7D"]
    Q3 --> R3["Δ_mass: 1D"]
    
    S["s0: batch×14"] --> T["Δ-State Reconstruction<br/>Set#1: s = s0 + Δs"]
    R1 --> T
    R2 --> T
    R3 --> T
    T --> U["Quaternion Normalization<br/>Set#1"]
    U --> V["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style S fill:#e1f5ff,color:#000000
    style D fill:#ffccbc,color:#000000
    style I fill:#ffccbc,color:#000000
    style J fill:#fff9c4,color:#000000
    style M fill:#fff9c4,color:#000000
    style P fill:#fff9c4,color:#000000
    style Q1 fill:#f8bbd0,color:#000000
    style Q2 fill:#f8bbd0,color:#000000
    style Q3 fill:#f8bbd0,color:#000000
    style V fill:#c8e6c9,color:#000000
```

## Direction C3: Enhanced C2 with RMSE Reduction Solutions

C2 architecture enhanced with 6 architectural solutions for RMSE reduction.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures<br/>8 frequencies"]
    C["context: batch×7"] --> D["DeepContextEncoder<br/>Set#2: 7→64→128→128→64→32"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×32"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×32"]
    F --> G
    G --> H["Input Projection<br/>49→128"]
    H --> I["Shared Stem<br/>Transformer Encoder<br/>4 layers, 4 heads<br/>d_model=128"]
    I --> J["shared_emb: batch×N×128"]
    
    J --> K["Enhanced z0 Derivation<br/>Solution 6: Hybrid Physics+Data"]
    K --> L["z0 Projection<br/>128→64"]
    L --> M["z0: batch×64"]
    
    J --> N["Full Sequence"]
    M --> O["Latent ODE Block<br/>Solution 4: RK4 Integration"]
    N --> O
    O --> P["z_traj: batch×N×64"]
    
    P --> Q1["PhysicsAwareTranslationBranch<br/>Solution 3: Explicit Physics"]
    P --> Q2["RotationBranchMinimal<br/>Solution 2: Minimal Rotation"]
    P --> Q3["MonotonicMassBranch<br/>Solution 1: Mass Monotonicity"]
    
    Q1 --> R1["Δ_translation: 6D"]
    Q2 --> R2["Δ_rotation: 7D"]
    Q3 --> R3["Δ_mass: 1D"]
    
    S["s0: batch×14"] --> T["Coordinated Branches<br/>Solution 5: Cross-Branch Coordination"]
    R1 --> T
    R2 --> T
    R3 --> T
    T --> U["Δ-State Reconstruction<br/>Set#1: s = s0 + Δs"]
    U --> V["Quaternion Normalization<br/>Set#1"]
    V --> W["state: batch×N×14"]
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style S fill:#e1f5ff,color:#000000
    style D fill:#ffccbc,color:#000000
    style I fill:#ffccbc,color:#000000
    style J fill:#fff9c4,color:#000000
    style K fill:#fff9c4,color:#000000
    style M fill:#fff9c4,color:#000000
    style O fill:#fff9c4,color:#000000
    style P fill:#fff9c4,color:#000000
    style Q1 fill:#f8bbd0,color:#000000
    style Q2 fill:#f8bbd0,color:#000000
    style Q3 fill:#f8bbd0,color:#000000
    style T fill:#e1bee7,color:#000000
    style W fill:#c8e6c9,color:#000000
```

## Direction D: Dependency-Aware Backbone

Single shared MLP backbone with ordered heads (mass → attitude → translation) to preserve physics dependencies without latent ODE blocks.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["FourierFeatures<br/>8 freq → 17D"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→32, GELU"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×N×32"]
    E --> G["Concatenate<br/>t_emb || ctx_emb"]
    F --> G
    G --> H["Shared Backbone<br/>MLP 4×256"]
    H --> I["G3 Mass Head<br/>[128,64]→1"]
    I --> J["m_pred"]
    
    %% Attitude Head
    H --> K1["latent copy"]
    J --> K2["m copy"]
    K1 --> L1
    K2 --> L1
    L1["Concat latent || m"] --> M["G2 Attitude Head<br/>[256,128,64]→7"]
    M --> N["normalize_quaternion"]
    
    %% Translation Head
    H --> P1["latent copy₂"]
    J --> P2["m copy₂"]
    N --> P3["q_norm, w"]
    P1 --> Q["Concat latent || m || q || w"]
    P2 --> Q
    P3 --> Q
    Q --> R["G1 Translation Head<br/>[256,128,128,64]→6"]
    
    %% Assemble state
    R --> S["translation"]
    N --> T["rotation + ω"]
    J --> U["mass"]
    S --> V["state: batch×N×14"]
    T --> V
    U --> V

    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style H fill:#fff9c4,color:#000000
    style I fill:#ffecb3,color:#000000
    style P fill:#ffe0b2,color:#000000
    style X fill:#ffccbc,color:#000000
    style Z fill:#c8e6c9,color:#000000
```

## Direction D1: Physics-Aware Dependency Backbone

Extends Direction D with physics feature injection, 6D rotation representation, and causal RK4 integration to reconstruct translation from predicted accelerations.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["FourierFeatures<br/>8 freq"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→32"]
    B --> E["t_emb"]
    D --> F["ctx_emb"]
    E --> G["Concat"]
    F --> G
    G --> H["Shared Backbone<br/>4×256"]
    
    %% Mass head
    H --> I["G3 Mass Head"] --> J["m_pred"]
    
    %% Physics features
    G --> K["Context broadcast"]
    A --> L["t grid"]
    L --> M["Altitude / velocity estimate"]
    M --> N["Physics Layer<br/>density, q_dyn, aero"]
    K --> N
    N --> O["physics_feat"]
    
    %% Attitude head with 6D rotation
    H --> P["latent copy"]
    J --> Q
    O --> R
    P --> S["Concat latent || m || physics"]
    Q --> S
    R --> S
    S --> T["G2 Head<br/>6D rot + ω"]
    T --> U["6D → quaternion"]
    
    %% Acceleration head + integrator
    H --> V["latent copy₂"]
    J --> W
    U --> X["rotation"]
    T --> Y["angular velocity"]
    O --> Z
    V --> AA["Concat latent || m || R || ω || physics"]
    W --> AA
    X --> AA
    Y --> AA
    Z --> AA
    AA --> AB["G1 Accel Head"]
    AB --> AC["Temporal Integrator<br/>RK4"]
    AC --> AD["translation (z,v)"]
    
    %% Assemble final state
    AD --> AE
    U --> AE
    Y --> AE
    J --> AE
    AE["state: batch×N×14"]

    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style H fill:#fff9c4,color:#000000
    style I fill:#ffecb3,color:#000000
    style T fill:#ffe0b2,color:#000000
    style AB fill:#ffccbc,color:#000000
    style AC fill:#d1c4e9,color:#000000
    style AE fill:#c8e6c9,color:#000000
```

## Direction D1.5: Soft-Physics Dependency Backbone

Adds light physics hints and optional structural constraints while keeping Direction D’s direct state predictions (no explicit integrator).

```mermaid
graph TD
    A["t: batch×N×1"] --> B["FourierFeatures<br/>8 freq → 17D"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→32, GELU"]
    B --> E["t_emb"]
    D --> F["ctx_emb"]
    E --> G["Concat"]
    F --> G
    G --> H["Shared Backbone<br/>MLP 4×256"]

    H --> I["G3 Mass Head"]
    C --> J["m0 context"] 
    I --> K["m_pred"]
    J --> K
    K --> L["Monotonic optional<br/>cumsum(-softplus)"]

    H --> M["latent copy"]
    L --> N["m copy"]
    M --> O
    N --> O
    O --> P["G2 Head<br/>6D rot (optional) + ω"]
    P --> Q["6D→quat || normalize q"]

    H --> R["latent copy₂"]
    L --> S
    Q --> T
    P --> U
    R --> V["Concat latent || m || q || ω"]
    S --> V
    T --> V
    U --> V
    V --> W["G1 Translation Head"]

    W --> X["translation"]
    Q --> Y["rotation"]
    U --> Z["angular vel"]
    L --> AA["mass"]
    X --> AB["state"]
    Y --> AB
    Z --> AB
    AA --> AB

    style H fill:#fff9c4,color:#000000
    style I fill:#ffecb3,color:#000000
    style P fill:#ffe0b2,color:#000000
    style W fill:#ffccbc,color:#000000
    style AB fill:#c8e6c9,color:#000000
```

## Direction D1.5.3: V2 Dataloader and Loss Function

Same architecture as Direction D1.5, but uses v2 dataloader (T_mag, q_dyn) and adjusted loss weights. Models receive v2 features but currently ignore them (future: v2 model versions will use InputBlockV2).

```mermaid
graph TD
    A["t: batch×N×1"] --> B["FourierFeatures<br/>8 freq → 17D"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→32, GELU"]
    E["T_mag: batch×N"] --> F["V2 Dataloader<br/>(loaded but unused)"]
    G["q_dyn: batch×N"] --> F
    B --> H["t_emb: batch×N×17"]
    D --> I["ctx_emb: batch×N×32"]
    H --> J["Concatenate<br/>t_emb || ctx_emb"]
    I --> J
    J --> K["Shared Backbone<br/>MLP 4×256"]
    
    K --> L["G3 Mass Head<br/>[128,64]→1"]
    C --> M["m0 context"]
    L --> N["m_pred"]
    M --> N
    N --> O["Monotonic<br/>cumsum(-softplus)"]
    
    K --> P["latent copy"]
    O --> Q["m copy"]
    P --> R
    Q --> R
    R --> S["G2 Head<br/>6D rot + ω"]
    S --> T["6D→quat || normalize q"]
    
    K --> U["latent copy₂"]
    O --> V
    T --> W
    S --> X
    U --> Y["Concat latent || m || q || ω"]
    V --> Y
    W --> Y
    X --> Y
    Y --> Z["G1 Translation Head<br/>[256,128,128,64]→6"]
    
    Z --> AA["translation"]
    T --> AB["rotation"]
    X --> AC["angular vel"]
    O --> AD["mass"]
    AA --> AE["state: batch×N×14"]
    AB --> AE
    AC --> AE
    AD --> AE
    
    F -.->|"Future: InputBlockV2"| K
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style E fill:#e1f5ff,color:#000000
    style G fill:#e1f5ff,color:#000000
    style F fill:#ffebee,color:#000000
    style K fill:#fff9c4,color:#000000
    style L fill:#ffecb3,color:#000000
    style S fill:#ffe0b2,color:#000000
    style Z fill:#ffccbc,color:#000000
    style AE fill:#c8e6c9,color:#000000
```

## Direction AN (Baseline): Shared Stem + Mission Branches + Physics Residuals

Shared stem with residual MLP, independent mission branches predicting full translation state, and physics residual layer that computes full ODE residuals.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures<br/>8 frequencies"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→128, Tanh"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×128"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×128"]
    F --> G
    G --> H["Concatenate<br/>t_emb || ctx_emb"]
    H --> I["ANSharedStem<br/>Residual MLP<br/>4 layers × 128<br/>Tanh, LayerNorm"]
    I --> J["latent: batch×N×128"]
    
    J --> K1["TranslationBranch<br/>128→128→128→6"]
    J --> K2["RotationBranch<br/>128→256→256→7"]
    J --> K3["MassBranch<br/>128→64→1"]
    
    K1 --> L1["translation: 6D<br/>[x,y,z,vx,vy,vz]"]
    K2 --> L2["rotation: 7D<br/>[q0,q1,q2,q3,wx,wy,wz]"]
    K3 --> L3["mass: 1D<br/>[m]"]
    
    L2 --> M["Quaternion Normalization"]
    L1 --> N["Pack State"]
    M --> N
    L3 --> N
    N --> O["state: batch×N×14"]
    
    O --> P["Physics Residual Layer<br/>Full ODE residual<br/>compute_dynamics"]
    A --> P
    Q["control: batch×N×4 (zeros)"] --> P
    P --> R["PhysicsResiduals<br/>diagnostics + loss"]
    
    O --> S["Output state"]
    R --> S
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style Q fill:#e1f5ff,color:#000000
    style I fill:#fff9c4,color:#000000
    style J fill:#fff9c4,color:#000000
    style K1 fill:#f8bbd0,color:#000000
    style K2 fill:#f8bbd0,color:#000000
    style K3 fill:#f8bbd0,color:#000000
    style P fill:#d1c4e9,color:#000000
    style O fill:#c8e6c9,color:#000000
    style S fill:#c8e6c9,color:#000000
```

## Direction AN1: Shared Stem + Physics-Guided Surrogate

Refinement of AN with reduced translation head, x/y integration, monotonic mass branch, and physics residuals used in a vertical-only loss.

```mermaid
graph TD
    A["t: batch×N×1"] --> B["TimeEmbedding<br/>FourierFeatures<br/>8 frequencies"]
    C["context: batch×7"] --> D["ContextEncoder<br/>7→128, Tanh"]
    B --> E["t_emb: batch×N×17"]
    D --> F["ctx_emb: batch×128"]
    E --> G["Broadcast ctx_emb<br/>to batch×N×128"]
    F --> G
    G --> H["Concatenate<br/>t_emb || ctx_emb"]
    H --> I["ANSharedStem<br/>Residual MLP<br/>4 layers × 128<br/>Tanh, LayerNorm"]
    I --> J["latent: batch×N×128"]
    
    %% Mission branches
    J --> K1["TranslationBranchReducedXYFree<br/>128→128→128→4<br/>[vx,vy,z,vz]"]
    J --> K2["RotationBranch<br/>128→256→256→7"]
    J --> K3["MonotonicMassBranch<br/>128→64→1<br/>m(t) ↓"]
    
    %% Split outputs
    K1 --> Lvxvy["vx,vy,z,vz"]
    K2 --> Lrot["[q0,q1,q2,q3,wx,wy,wz]"]
    K3 --> Lm["m(t)"]
    
    %% x,y integration from velocities
    Lvxvy --> INT["Kinematic Integration<br/>x(0)=y(0)=0<br/>x,y = ∫ v_x,v_y dt"]
    A --> INT
    INT --> Lpos["x,y,z"]
    
    %% Pack state
    Lrot --> M["Quaternion Normalization"]
    Lpos --> N["Pack State"]
    M --> N
    Lm --> N
    N --> O["state: batch×N×14"]
    
    %% Physics residuals (full) and vertical-only loss
    O --> P["PhysicsResidualLayer<br/>compute_dynamics"]
    A --> P
    Q["control: batch×N×4 (zeros)"] --> P
    P --> R["PhysicsResiduals<br/>diagnostics"]
    
    %% Training loss (conceptual)
    O --> LVERT["PINNLoss / PINNLossV2<br/>Vertical-only residual<br/>T_eff from ṁ"]
    C --> LVERT
    
    O --> S["Output state"]
    R --> S
    
    style A fill:#e1f5ff,color:#000000
    style C fill:#e1f5ff,color:#000000
    style Q fill:#e1f5ff,color:#000000
    style I fill:#fff9c4,color:#000000
    style J fill:#fff9c4,color:#000000
    style K1 fill:#f8bbd0,color:#000000
    style K2 fill:#f8bbd0,color:#000000
    style K3 fill:#f8bbd0,color:#000000
    style INT fill:#e3f2fd,color:#000000
    style P fill:#d1c4e9,color:#000000
    style LVERT fill:#ffe0b2,color:#000000
    style O fill:#c8e6c9,color:#000000
    style S fill:#c8e6c9,color:#000000
```

## Architecture Comparison Summary

| Architecture | Key Features | RMSE (Typical) |
|--------------|--------------|----------------|
| **Baseline** | Vanilla MLP, Fourier features, shallow context encoder | ~0.9 |
| **Direction A** | Latent ODE, encode-evolve-decode | ~0.85-0.9 |
| **Direction B** | Transformer sequence model | ~0.86 |
| **Direction C** | Hybrid: Transformer z0 + Latent ODE | ~0.9 |
| **Direction C1** | C + Set#1 (stability) + Set#2 (deep context) | ~0.84 |
| **Direction C2** | C1 + Shared Stem + Dedicated Branches | ~0.96 |
| **Direction C3** | C2 + 6 RMSE reduction solutions | Target: 0.60-0.75 |
| **Direction D** | Dependency-aware backbone with ordered heads | ~0.30 |
| **Direction D1** | Direction D + physics features + RK4 integration | ~0.28-0.29 |
| **Direction D1.5** | D + soft physics + mass monotonicity + 6D rotation | ~0.20 |
| **Direction D1.5.3** | D1.5 + v2 dataloader (T_mag, q_dyn) + v2 loss | **0.198** |
| **Direction D1.5.4** | D1.5 + central diff (PINNLossV2) | **0.254** |
| **Direction AN** | Shared stem + mission branches + full ODE residual | **0.197** |
| **Direction AN1** | AN + x/y integration, monotonic mass, vertical-only residual | **0.197** |

## Component Relationships

```mermaid
graph LR
    A["Baseline PINN"] --> B["Direction A<br/>Latent ODE"]
    A --> C["Direction B<br/>Transformer"]
    B --> D["Direction C<br/>Hybrid"]
    C --> D
    D --> E["Direction C1<br/>+ Set#1 + Set#2"]
    E --> F["Direction C2<br/>+ Shared Stem + Branches"]
    F --> G["Direction C3<br/>+ 6 Solutions"]
    A --> H["Direction D<br/>Dependency Backbone"]
    H --> I["Direction D1<br/>Physics-Aware"]
    H --> J["Direction D1.5<br/>Soft Physics"]
    J --> K["Direction D1.5.3<br/>V2 Dataloader"]
    K --> K2["Direction D1.5.4<br/>Central Diff"]
    A --> L["Direction AN<br/>Shared Stem + Branches (Baseline)"]
    L --> M["Direction AN1<br/>Physics-Guided AN"]
    
    style A fill:#e1f5ff,color:#000000
    style B fill:#fff9c4,color:#000000
    style C fill:#ffccbc,color:#000000
    style D fill:#f8bbd0,color:#000000
    style E fill:#e1bee7,color:#000000
    style F fill:#c8e6c9,color:#000000
    style G fill:#a5d6a7,color:#000000
    style H fill:#b3e5fc,color:#000000
    style I fill:#90caf9,color:#000000
    style J fill:#81d4fa,color:#000000
    style K fill:#64b5f6,color:#000000
    style K2 fill:#4fc3f7,color:#000000
    style L fill:#ce93d8,color:#000000
    style M fill:#ba68c8,color:#000000
```

## Data Flow Summary

All architectures follow the same input/output interface:

```
Input:  (t: [batch, N, 1], context: [batch, 7])
Output: state: [batch, N, 14]
```

Where:
- `t`: Nondimensional time grid
- `context`: Normalized context parameters (m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag)
- `state`: 14D state vector [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]

## References

- **Architecture Changelog**: [ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md) - Detailed implementation history
- **Thesis Notes**: [thesis_notes.md](thesis_notes.md) - Research notes and development decisions
- **C3 Implementation Guide**: [expANAL_SOLS.md](expANAL_SOLS.md) - C3 architecture details and solutions
