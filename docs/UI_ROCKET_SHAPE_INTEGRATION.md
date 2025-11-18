### UI integration guide: Rocket shape design and simulation

This guide explains how a UI can let users design the rocket "shape" and run the ascent simulation without modifying C++ code.

### Audience

- UI engineer building controls that map to the simulator’s aerodynamic and geometry parameters.

### What you will interact with

- Runtime config your UI writes: a JSON file (any path)
- Simulator executable your UI runs: `validate_dynamics.exe` (built from `tests/validate_dynamics.cpp`)
- Output your UI reads/visualizes: `trajectory.csv`

### Relevant code and parameters

- `src/physics/dynamics.hpp` → `AscentDynamics::Params`
  - `A` reference area in m²
  - `set_reference_area_from_diameter(double diameter_m)` convenience
  - `aero` → `physics::aerodynamics::AeroParams` (Cd vs Mach)
  - Other: `enable_wind`, `earth.latitude`, `Isp`, `Tmax`, `m_dry`, `m_prop`
- `src/physics/aerodynamics/aerodynamics.hpp` → `AeroParams`
  - `Cd_subsonic`, `Cd_transonic_peak`, `Cd_supersonic`, `mach_transonic_start`, `mach_transonic_end`, `Cd_alpha_slope`
- `tests/validate_dynamics.cpp`
  - Accepts `--config path.json`, applies parameters, integrates, writes `trajectory.csv`

### UI inputs to expose

- Geometry
  - `diameter_m` (m). UI can optionally expose direct `A` override.
- Aerodynamics (simple Cd(Mach) curve)
  - `aero_Cd_subsonic`, `aero_Cd_transonic_peak`, `aero_Cd_supersonic`
  - `aero_mach_transonic_start`, `aero_mach_transonic_end`
  - `aero_Cd_alpha_slope` (leave 0 in planar model)
- Vehicle/environment basics
  - `Isp`, `Tmax`, `m_dry`, `m_prop`, `enable_wind`, `latitude_deg`

### JSON your UI should write (flat schema)

- Keys consumed by the simulator:
  - `diameter_m`, `A` (optional direct area)
  - `Isp`, `Tmax`, `m_dry`, `m_prop`, `enable_wind`, `latitude_deg`
  - `aero_Cd_subsonic`, `aero_Cd_transonic_peak`, `aero_Cd_supersonic`
  - `aero_mach_transonic_start`, `aero_mach_transonic_end`, `aero_Cd_alpha_slope`

#### Example config.json

```json
{
  "diameter_m": 1.8,
  "Isp": 305,
  "Tmax": 200000,
  "m_dry": 1500,
  "m_prop": 4500,
  "enable_wind": false,
  "latitude_deg": 28.5,
  "aero_Cd_subsonic": 0.32,
  "aero_Cd_transonic_peak": 1.15,
  "aero_Cd_supersonic": 0.75,
  "aero_mach_transonic_start": 0.8,
  "aero_mach_transonic_end": 1.2,
  "aero_Cd_alpha_slope": 0.0
}
```

### How to run the simulator from the UI

1) Ensure the project is built so `validate_dynamics.exe` exists (e.g., `build/Release/validate_dynamics.exe`).

2) Spawn the process with the config path:

- Windows example:
  - `build\Release\validate_dynamics.exe --config C:\\path\\to\\config.json`

3) Read `trajectory.csv` from the working directory and plot/show:

- Columns: `t,x,y,vx,vy,m,q,mach`

### Control-to-parameter mapping (suggested UI)

- Diameter slider → `diameter_m` (or switch to direct `A` input)
- Cd curve presets/knobs → the 5 aero fields above
- Toggles → `enable_wind`
- Launch site latitude → `latitude_deg`
- Vehicle inputs → `m_dry`, `m_prop`, `Tmax`, `Isp`

### Direct C++ API option (if embedding)

- Construct `AscentDynamics::Params`, call `set_reference_area_from_diameter`, tweak `p.aero`, and run `ForwardIntegrator::integrate_rk45_with_staging(...)`. Parse results in-app instead of using the CSV.


