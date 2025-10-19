## Title
Physics‑Informed Neural Network for **6‑DOF** Rocket Trajectory Optimization — End‑to‑End Project Guide

## Purpose
A step‑by‑step, production‑style plan to build, evaluate, and report a 6‑DOF PINN surrogate embedded in a constrained optimizer for ascent guidance. This version upgrades dynamics to full 6‑DOF and **includes wind and q‑limit** in the core scope. **Staging** remains an extension.

---

## 0) Scope, success criteria, and assumptions
- **Scope (core)**: Single‑stage 6‑DOF rigid body, quaternion attitude, simple aero (CL/CD/Cm) about a nominal axis, constant or altitude‑varying gravity, **wind**, and **dynamic pressure (q) limit**. Thrust is gimbaled with magnitude/angle/rate limits.
- **Out‑of‑scope (extensions)**: staging, flexible body, detailed aero tables (Mach/Re grids), 3‑D winds, Earth rotation, planetary curvature beyond standard gravity model.
- **Success criteria** (evaluate against a direct collocation baseline on identical hardware):
  1) **Accuracy**: time‑averaged RMSE(state) ≤ 1–2% of scale; terminal objective gap |ΔJ| ≤ 2%.
  2) **Feasibility**: post‑projection IPOPT polish yields ≤ 1% path‑constraint violations; q‑spikes removed.
  3) **Speed**: ≥ **10×** faster wall‑clock for PINN‑based optimization vs. re‑solving collocation.
  4) **Robustness**: under ±10% in {Cd, Isp} and ±20% wind, terminal dispersion within predefined bounds.
- **Units**: SI. Provide a **nondimensionalization** table and consistent scales for OCP, PINN, and plots.

---

## 1) Architecture overview
```
repo/
  src/                 # C++ core physics + integrators
    physics/
    ocp/
  python/              # Data gen, training, analysis
    data_gen/
    training/
    eval/
  configs/             # YAML configs (phys, limits, experiments)
  tests/               # GoogleTest (C++), pytest (Python)
  notebooks/           # Exploration & figures
  docs/                # This guide + API docs
  CMakeLists.txt
```

Key components:
- **Physics core (C++)**: 6‑DOF RHS, wind hooks, q‑monitor, RK4 + RK45 (odeint), batch RHS for later LibTorch.
- **OCP baseline (CasADi)**: direct collocation (Hermite–Simpson or LGR) with IPOPT.
- **Dataset generator (Py)**: calls OCP across parameter grid; exports HDF5/NPZ (+ metadata: seeds, scales, config hash).
- **PINN training (PyTorch/JAX)**: solution‑network or residual‑Δ model; AD‑based physics residuals; uncertainty via ensemble.
- **Optimizer via PINN**: gradient‑based control knot optimization through the surrogate; optional feasibility polish via IPOPT.

---

## 2) Nondimensionalization & scaling (do this first)
Create `configs/scales.yaml` and a helper `python/common/scales.py`.

Example scales: position `L=10^4 m`, velocity `V=√(g0 L)`, time `T=L/V`, mass `M=m0`, thrust `F=M g0`, angular rate `Ω=1/T`, moments `N=F·ref_len`. Provide `to_nd()` and `from_nd()` for states, controls, params.

Why: improves conditioning for OCP and stabilizes PINN losses/gradients.

---

## 3) 6‑DOF dynamics model
**State** (14 vars): `r_i∈R^3` (position, inertial), `v_i∈R^3` (velocity, inertial), `q_bi∈H` (unit quaternion body→inertial), `ω_b∈R^3` (body rates), `m` (mass).

**Controls**: thrust magnitude `T`, gimbal direction in body `{θ_g, φ_g}` (or unit vector), plus optional torque controls if RCS modeled (default: thrust vector produces both force and moment via offset or gimbal).

**Forces & moments**
- Gravity: `g_i(y)` (constant or inverse‑square).
- Aerodynamic: dynamic pressure `q = 0.5 ρ ||v_rel||^2`, with air‑relative velocity `v_rel = v_i − w_i(y,t)`. Transform to body: `v_b = R_i2b(q) v_rel`. Use simple coefficient model:
  - Drag: `F_D_b = − q S C_D(α,|v|) * v̂_b`.
  - Lift: `F_L_b = q S C_Lα * α * ê_L` (small‑angle model).
  - Moment: `M_b = q S l_ref [ C_mα α, C_lβ β, C_nβ β ]` (diagonal small‑disturbance model). Keep coefficients in `configs/aero.yaml`.
- Thrust: `F_T_b = T * û_gimbal`, `M_T_b` from gimbal offset/engine cardan if desired (optional in core).

**Equations**
- kinematics: `ṙ_i = v_i`.
- translation: `v̇_i = (1/m) [ R_b2i F_b ] + g_i` where `F_b = F_T_b + F_Aero_b`.
- attitude: `q̇ = 0.5 Ω(ω_b) q`, `Ω(ω)` the quaternion rate matrix; renormalize periodically.
- rotation: `ω̇_b = I^{-1} ( M_b − ω_b × (I ω_b) )` with body inertia `I`.
- mass: `ṁ = − T/(Isp g0)`.

**Constraints & monitors**
- Thrust bounds & rate limits; gimbal angle limits & rates; `m ≥ m_dry` clamp; **q‑limit**: `q ≤ q_max` (hard in OCP, monitored in sim).

---

## 4) Baseline OCP (direct collocation)
- Transcription: **Hermite–Simpson** (start) with path constraints (q‑limit, actuator limits).
- Variables: state knots + control knots (thrust & gimbal).
- Objective examples: minimize propellant; maximize apogee subject to fuel; track target MECO state.
- Solver: IPOPT via CasADi. Record tolerances, mesh size, and scaling in a single YAML.
- Output: optimal knots, integrated truth trajectory on a standard grid.

---

## 5) Data generation
- Sample `(m0, Cd, CLα, Cmα, Isp, T_max, wind scale)` via **Latin Hypercube**; fix train/val/test seeds.
- For each sample: solve OCP → integrate → export `{states, controls, params, monitors}` on uniform time grid + metadata.
- Store in **HDF5/NPZ** with a checksum of configs and git hash.

---

## 6) PINN design & training
- **Inputs**: time `t` (Fourier features), param/context vector `p` (scaled), optional low‑dim control parameterization (control knots or basis coefficients).
- **Targets**: state trajectory (or Δ‑state residual vs coarse integrator).
- **Loss**: `L = λ_data L_data + λ_phys L_phys + λ_bc L_bc (+ λ_ctrl L_ctrl)`.
  - `L_phys` uses AD through the network to build residuals of the **scaled** ODE.
  - **Boundary**: initial state, terminal soft constraints.
  - **Scheduling**: start with data‑heavy (λ_phys small) → ramp λ_phys.
- **Architecture**: MLP 6×256 (tanh) + Fourier features; alt: SIREN for time; residual/skip connections; layer norm.
- **Stability tricks**: quaternion **unit‑norm projection** in the output; **softplus** for mass positivity w/ `m = m_dry + softplus(ŷ_m)`; control squashing (`sigmoid` / `tanh`) to enforce bounds.
- **UQ**: ensemble (K=5) with different seeds; compute coverage.

---

## 7) Optimizing through the PINN
- Parameterize control as B‑spline knots (magnitude + 2‑angle). Optimize knots with Adam/L‑BFGS using gradients from PINN; include soft penalties for q‑limit (or differentiable barrier).
- **Projection polish**: run a short IPOPT solve initialized at the PINN solution to measure feasibility/optimality gap.

---

## 8) Experiments & metrics
1) **Held‑out accuracy**: RMSE over states; terminal objective gap; attitude error via quaternion distance.
2) **Optimizer comparison**: wall‑clock, iterations, objective, violations.
3) **Robustness**: parameter sweeps; wind gust profiles; report dispersion & violations.
4) **Ablations**: λ_phys schedule, residual vs pure PINN, network size, data quantity, control basis.
5) **UQ calibration**: coverage of 50/90% intervals.

---

## 9) Reproducibility & logging
- One **Hydra/YAML** config per experiment; log **git hash**, seeds, library versions, machine spec.
- CI smoke run: minimal OCP + tiny training + 2 plots.

---

## 10) Timeline (5 months)
- **M1**: finalize model & scaling; implement 6‑DOF C++; write unit tests; validate against analytic/simple cases.
- **M2**: OCP baseline with q‑limit; data generator; first small dataset.
- **M3**: PINN prototype; stabilize training; implement residual‑Δ variant; early accuracy eval.
- **M4**: Optimizer via PINN; full experiments; robustness & UQ; ablations.
- **M5**: Paper/thesis; figures; reproducibility package; presentation.

---

## 11) Deliverables
- C++ physics lib + tests; CasADi OCP scripts; datasets; training code; trained weights; evaluation notebooks; final report + reproducibility script.

---

## 12) Risk log & mitigations
- PINN instability → residual‑Δ, curriculum on λ_phys, stronger scaling; unit‑norm quaternion projection.
- OCP infeasibility under q‑limit → soften via slacks then tighten; warm‑start strategy; mesh refinement.
- Data leakage/splits → strict seeds + metadata validation; hash configs.
