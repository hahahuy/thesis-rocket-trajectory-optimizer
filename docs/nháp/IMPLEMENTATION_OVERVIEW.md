## Goal
Stand up the development environment and implement the first, well‑tested 6‑DOF physics module with wind & q‑monitoring, ready for OCP and PINN.

---

## A) Tooling & libraries
### System
- **CMake ≥ 3.21**, **C++17**
- **Eigen 3.4** (linear algebra)
- **Boost.Odeint** (RK45) + custom RK4
- **spdlog** (logging), **fmt** (formatting)
- **yaml‑cpp** (configs)
- **GoogleTest** (C++ unit tests)
- **CasADi (Python)** for OCP baseline (`pip install casadi`)
- **Python 3.10+** with: numpy, scipy, matplotlib, hydra‑core, h5py, pytorch or jax, wandb/tensorboard (optional)

### Repo bootstrap
```
mkdir -p src/physics src/ocp tests python/{data_gen,training,eval,common} configs docs notebooks bin
```

`CMakeLists.txt` (top‑level) should:
- find Eigen, Boost
- add `src` as a library target `rocket_physics`
- add executable `run_dynamics_demo`
- enable tests (GoogleTest)

---

## B) Config files
Create `configs/phys.yaml`:
```yaml
phys:
  Cd: 0.3
  CL_alpha: 3.5    # 1/rad, small‑angle lift slope
  Cm_alpha: -0.8   # 1/rad, pitch moment slope
  S: 0.05          # m^2, ref area
  l_ref: 1.2       # m, ref length
  I_body: [12.0, 12.0, 2.5]   # kg·m^2, diag inertia (Ix,Iy,Iz)
  Isp: 250.0
  rho0: 1.225
  H: 8500.0
  g0: 9.80665
limits:
  Tmax: 4000.0     # N
  mdry: 35.0       # kg
  qmax: 40_000.0   # Pa
  gimbal_max_deg: 7.0
  thrust_rate: 500.0  # N/s
  gimbal_rate_deg: 10.0 # deg/s
wind:
  type: constant
  u: 5.0   # m/s east
  v: 0.0   # m/s north (for planar simplification use x only)
```

`configs/scales.yaml` (example):
```yaml
scales:
  L: 10000.0  # m
  V: 313.0    # m/s (≈ sqrt(g0*L))
  T: 32.0     # s (≈ L/V)
  M: 50.0     # kg
  F: 490.0    # N (M*g0)
  W: 1.0      # rad/s (≈ 1/T)
```

---

## C) C++ API design (headers only sketch)
**`src/physics/types.hpp`**
```cpp
#pragma once
#include <Eigen/Dense>
#include <functional>

namespace phys6d {
  using Vec3 = Eigen::Vector3d; using Mat3 = Eigen::Matrix3d;

  struct State {                 // 14 vars total
    Vec3 r_i;    // position (inertial)
    Vec3 v_i;    // velocity (inertial)
    Eigen::Quaterniond q_bi; // body->inertial
    Vec3 w_b;    // body angular rates
    double m;    // mass
  };

  struct Control {
    double T;     // thrust N
    Vec3 uT_b;    // unit thrust dir in body (normalized)
  };

  struct Phys { double Cd, CL_alpha, Cm_alpha, S, l_ref; Vec3 Idiag; double Isp, rho0, H, g0; };
  struct Limits { double Tmax, mdry, qmax; double gimbal_max_rad, thrust_rate, gimbal_rate_rad; };

  using GravityFunc = std::function<double(double y)>; // g(y)
  using WindFunc = std::function<Vec3(double y, double t)>; // inertial wind

  struct Env { GravityFunc g; WindFunc wind; };

  struct Diag { double rho=0, q=0, g=0; bool q_violation=false, m_under=false; };
}
```

**`src/physics/frames.hpp`** (quaternion utilities)
```cpp
#pragma once
#include "types.hpp"
namespace phys6d {
  inline Mat3 R_b2i(const Eigen::Quaterniond& q){ return q.toRotationMatrix(); }
  inline Mat3 R_i2b(const Eigen::Quaterniond& q){ return q.conjugate().toRotationMatrix(); }
  inline Eigen::Quaterniond quat_renorm(const Eigen::Quaterniond& q){ return q.normalized(); }
}
```

**`src/physics/dynamics.hpp`**
```cpp
#pragma once
#include "types.hpp"
namespace phys6d {
  State rhs(const State& s, const Control& u,
            const Phys& P, const Limits& L, const Env& E,
            double t, Diag* d=nullptr);

  // Batch form for later LibTorch port
  void rhs_batch(const Eigen::Ref<const Eigen::Matrix<double,14,Eigen::Dynamic>>& X,
                 const Eigen::Ref<const Eigen::Matrix<double,4, Eigen::Dynamic>>& U, // T + dir(3)
                 const Phys& P, const Limits& L, const Env& E,
                 Eigen::Ref<Eigen::Matrix<double,14,Eigen::Dynamic>> dXdt);
}
```

**`src/physics/integrator.hpp`**
```cpp
#pragma once
#include "dynamics.hpp"
#include <vector>
namespace phys6d {
  using Time = double; using TSeries = std::vector<Time>; using XSeries = std::vector<State>;
  using ControlCB = std::function<Control(Time, const State&)>;

  XSeries integrate_rk4(const State& x0, const TSeries& tgrid,
                        const Phys& P, const Limits& L, const Env& E,
                        ControlCB ucb);

  XSeries integrate_rk45(const State& x0, Time t0, Time t1,
                         const Phys& P, const Limits& L, const Env& E,
                         ControlCB ucb,
                         double abs_tol=1e-6, double rel_tol=1e-6, double dt_init=1e-3);
}
```

---

## D) Dynamics implementation notes (`src/physics/dynamics.cpp`)
Key steps per call to `rhs`:
1. **Clamp**: `m = max(s.m, L.mdry)`, `T = clamp(u.T, 0, L.Tmax)`, `uT_b = normalized(u.uT_b)`.
2. **Wind & relative airspeed**: `w_i = E.wind(s.r_i.z(), t)`; `v_rel_i = s.v_i - w_i`; in body: `v_b = R_i2b(q) v_rel_i`; speed `V = max(||v_rel_i||, 1e-6)`.
3. **Atmosphere**: `rho = P.rho0 * exp(-max(0,z)/P.H)`; `q = 0.5*rho*V*V`.
4. **Aero coefficients** (small‑angle): angle of attack `α ≈ atan2(v_b.z(), v_b.x())`; sideslip `β ≈ asin(v_b.y()/V)`;
   - Drag dir in body: `−v_b.normalized()`
   - Lift dir: orthogonal to `v_b` in x‑z plane.
5. **Forces in body**:
   - `F_D_b = − q P.S P.Cd * v_b.normalized()`
   - `F_L_b =  q P.S P.CL_alpha * α * e_L`
   - `F_T_b =  T * uT_b`
   - `F_b = F_T_b + F_D_b + F_L_b`
6. **Moments in body** (small‑disturbance): `M_b = q P.S P.l_ref * [0, P.Cm_alpha*α, 0]` (pitch only to start). Extend with roll/yaw if needed.
7. **Translational accel**: `a_i = R_b2i(q) * (F_b / m) + [0,0,-E.g(z)]` (assuming z‑up inertial; adjust sign to your convention).
8. **Rotational accel**: using diagonal inertia `I = diag(P.Idiag)`: `ẇ = I^{-1}( M_b − w × (I w) )`.
9. **Attitude kinematics**: `q̇ = 0.5 * Ω(w_b) * q`; renormalize in integrator every step or N steps.
10. **Mass flow**: `ṁ = − T/(P.Isp * P.g0)`.
11. **Diagnostics**: fill `Diag` and set `q_violation = (q > L.qmax)`.

**Sign conventions**: choose **z‑up inertial**; document clearly (docs/frames.md). If you keep NED, flip signs consistently.

---

## E) Integrators
- RK4: fixed time grid; sample control at sub‑steps; normalize quaternion after each full step.
- RK45 (odeint): adaptive step; provide a state wrapper and stepper; re‑project quaternion in an observer callback.
- Both accept a `ControlCB` that can implement **rate limits** (wrap user control with a limiter that respects `thrust_rate`, `gimbal_rate`).

---

## F) Unit tests (`tests/test_dynamics.cpp`)
1. **Ballistic (no aero, no thrust)**: set `Cd=CLα=Cmα=0, wind=0`; check free‑fall `y(t)` and `vy(t)` against closed form for short horizon (≤ 1 s) within 1e‑6.
2. **Mass depletion**: constant thrust; check `m(t)` analytic.
3. **Hover sanity**: in vacuum (no aero, wind=0), with `T≈m g`, check `z` drift small.
4. **Quaternion norm**: stays within 1±1e‑9 over long run via renorm.
5. **q‑limit monitor**: inject high speed/wind → diagnostic `q_violation=true`.
6. **Batch vs scalar**: random states, ensure `rhs_batch` columns ≡ scalar `rhs` to 1e‑12.

---

## G) Demo program (`bin/run_dynamics_demo.cpp`)
- Load `configs/phys.yaml` & `scales.yaml`.
- Simulate a 30 s ascent with a simple open‑loop gimbal schedule.
- Write CSV: `t,r_i(vectors),v_i(vectors),q,ω_b,m,T,uT_b,rho,q_dyn`.

---

## H) Python glue for OCP & data gen (outline)
- `python/data_gen/solve_ocp.py`: CasADi model (scaled); Hermite–Simpson transcription; q‑limit as path constraint; export knots + truth trajectory.
- `python/data_gen/make_dataset.py`: LHS sampler; loop `solve_ocp`; store HDF5/NPZ + metadata (seeds, configs, git hash).

---

## I) Next steps
- Implement OCP baseline; generate a tiny dataset; verify demos/plots; then proceed to PINN prototype and training.

