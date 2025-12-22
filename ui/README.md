### Rocket Shape + Ascent Simulator UI (Python)

This desktop UI lets you design the rocket "shape" via diameter/area and simple Cd(Mach) parameters, writes the simulator config JSON, runs `validate_dynamics.exe`, and plots `trajectory.csv`.

### Prereqs
- Python 3.9+
- Built simulator executable: `validate_dynamics.exe` (typically at `build/Release/validate_dynamics.exe`)

### Install deps
```bash
pip install -r ui/requirements.txt
```

### Run the UI
```bash
python ui/app.py
```

### Usage
- Set `Simulator exe` to the built `validate_dynamics.exe`.
- Set `Working dir` (CSV will be written there as `trajectory.csv`).
- Choose a path for `config.json` (anywhere).
- Fill inputs according to `docs/UI_ROCKET_SHAPE_INTEGRATION.md`:
  - `diameter_m` or provide `A` override
  - Aerodynamics: `aero_Cd_subsonic`, `aero_Cd_transonic_peak`, `aero_Cd_supersonic`, `aero_mach_transonic_start`, `aero_mach_transonic_end`, `aero_Cd_alpha_slope`
  - Vehicle/environment: `Isp`, `Tmax`, `m_dry`, `m_prop`, `enable_wind`, `latitude_deg`
- Click "Save config.json" to write the JSON.
- Click "Run simulator" to execute. The app will parse and plot `trajectory.csv` with:
  - x-y ground track
  - speed vs time
  - Mach vs time

### CSV format expected
- Columns: `t,x,y,vx,vy,m,q,mach`

### Notes
- On Windows, the app launches the exe without an interactive console and captures stdout/stderr. Errors will be shown in a dialog.
- If you embed directly, see `docs/UI_ROCKET_SHAPE_INTEGRATION.md` for using the C++ API instead of CSV.
