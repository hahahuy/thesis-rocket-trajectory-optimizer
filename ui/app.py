import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


@dataclass
class UiState:
    exe_path: str = ""
    working_dir: str = ""
    config_path: str = ""


class LabeledEntry(tk.Frame):
    def __init__(self, master, label_text: str, default_value: str = "", width: int = 12):
        super().__init__(master)
        self.label = tk.Label(self, text=label_text, anchor="w")
        self.entry = tk.Entry(self, width=width)
        self.entry.insert(0, default_value)
        self.label.pack(side=tk.LEFT, padx=(0, 8))
        self.entry.pack(side=tk.LEFT)

    def get(self) -> str:
        return self.entry.get().strip()

    def set(self, value: str) -> None:
        self.entry.delete(0, tk.END)
        self.entry.insert(0, value)


class RocketUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Rocket Shape + Ascent Simulator")
        self.state = UiState()

        # Top-level panes
        self.inputs_frame = tk.Frame(self)
        self.inputs_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Build input controls per docs/UI_ROCKET_SHAPE_INTEGRATION.md
        self._build_io_paths_section()
        self._build_geometry_section()
        self._build_aero_section()
        self._build_vehicle_env_section()
        self._build_actions_section()

        # Plot area
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax_xy = self.figure.add_subplot(2, 2, 1)
        self.ax_vt = self.figure.add_subplot(2, 2, 2)
        self.ax_mach_t = self.figure.add_subplot(2, 1, 2)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_io_paths_section(self) -> None:
        frame = tk.LabelFrame(self.inputs_frame, text="Paths")
        frame.pack(fill=tk.X, pady=(0, 10))

        self.entry_exe = LabeledEntry(frame, "Simulator exe:", "build/Release/validate_dynamics.exe", 32)
        self.entry_exe.pack(anchor="w", pady=2)
        btn_browse_exe = tk.Button(frame, text="Browse", command=self._browse_exe)
        btn_browse_exe.pack(anchor="w", pady=2)

        self.entry_workdir = LabeledEntry(frame, "Working dir:", str(Path.cwd()), 32)
        self.entry_workdir.pack(anchor="w", pady=2)
        btn_browse_workdir = tk.Button(frame, text="Select Folder", command=self._browse_workdir)
        btn_browse_workdir.pack(anchor="w", pady=2)

        self.entry_cfg = LabeledEntry(frame, "config.json:", str(Path.cwd() / "config.json"), 32)
        self.entry_cfg.pack(anchor="w", pady=2)
        btn_browse_cfg = tk.Button(frame, text="Save As...", command=self._browse_cfg)
        btn_browse_cfg.pack(anchor="w", pady=2)

    def _build_geometry_section(self) -> None:
        frame = tk.LabelFrame(self.inputs_frame, text="Geometry")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.in_diameter_m = LabeledEntry(frame, "diameter_m (m):", "1.8")
        self.in_diameter_m.pack(anchor="w", pady=2)
        self.in_A_override = LabeledEntry(frame, "A override (m^2):", "")
        self.in_A_override.pack(anchor="w", pady=2)

    def _build_aero_section(self) -> None:
        frame = tk.LabelFrame(self.inputs_frame, text="Aerodynamics (Cd vs Mach)")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.in_Cd_sub = LabeledEntry(frame, "Cd_subsonic:", "0.32")
        self.in_Cd_sub.pack(anchor="w", pady=2)
        self.in_Cd_peak = LabeledEntry(frame, "Cd_transonic_peak:", "1.15")
        self.in_Cd_peak.pack(anchor="w", pady=2)
        self.in_Cd_sup = LabeledEntry(frame, "Cd_supersonic:", "0.75")
        self.in_Cd_sup.pack(anchor="w", pady=2)
        self.in_mach_start = LabeledEntry(frame, "mach_transonic_start:", "0.8")
        self.in_mach_start.pack(anchor="w", pady=2)
        self.in_mach_end = LabeledEntry(frame, "mach_transonic_end:", "1.2")
        self.in_mach_end.pack(anchor="w", pady=2)
        self.in_alpha_slope = LabeledEntry(frame, "Cd_alpha_slope:", "0.0")
        self.in_alpha_slope.pack(anchor="w", pady=2)

    def _build_vehicle_env_section(self) -> None:
        frame = tk.LabelFrame(self.inputs_frame, text="Vehicle & Environment")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.in_Isp = LabeledEntry(frame, "Isp (s):", "305")
        self.in_Isp.pack(anchor="w", pady=2)
        self.in_Tmax = LabeledEntry(frame, "Tmax (N):", "200000")
        self.in_Tmax.pack(anchor="w", pady=2)
        self.in_m_dry = LabeledEntry(frame, "m_dry (kg):", "1500")
        self.in_m_dry.pack(anchor="w", pady=2)
        self.in_m_prop = LabeledEntry(frame, "m_prop (kg):", "4500")
        self.in_m_prop.pack(anchor="w", pady=2)

        row = tk.Frame(frame)
        row.pack(anchor="w", pady=2)
        tk.Label(row, text="enable_wind:").pack(side=tk.LEFT)
        self.var_wind = tk.BooleanVar(value=False)
        tk.Checkbutton(row, variable=self.var_wind).pack(side=tk.LEFT, padx=(6, 0))

        self.in_lat = LabeledEntry(frame, "latitude_deg:", "28.5")
        self.in_lat.pack(anchor="w", pady=2)

    def _build_actions_section(self) -> None:
        frame = tk.Frame(self.inputs_frame)
        frame.pack(fill=tk.X, pady=(10, 0))
        tk.Button(frame, text="Save config.json", command=self.on_save_config).pack(fill=tk.X)
        tk.Button(frame, text="Run simulator", command=self.on_run_sim).pack(fill=tk.X, pady=(6, 0))
        tk.Button(frame, text="Load config.json", command=self.on_load_config).pack(fill=tk.X, pady=(6, 0))

    # I/O helpers
    def _browse_exe(self) -> None:
        path = filedialog.askopenfilename(title="Select validate_dynamics.exe", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.entry_exe.set(path)

    def _browse_workdir(self) -> None:
        path = filedialog.askdirectory(title="Select working directory")
        if path:
            self.entry_workdir.set(path)

    def _browse_cfg(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], title="Save config.json")
        if path:
            self.entry_cfg.set(path)

    # Config handling
    def _collect_config(self) -> dict:
        cfg = {}
        def parse_float(entry: LabeledEntry, key: str, allow_empty: bool = False):
            value = entry.get()
            if value == "" and allow_empty:
                return
            try:
                cfg[key] = float(value)
            except ValueError:
                raise ValueError(f"Invalid float for {key}: '{value}'")

        parse_float(self.in_diameter_m, "diameter_m")
        parse_float(self.in_A_override, "A", allow_empty=True)
        parse_float(self.in_Isp, "Isp")
        parse_float(self.in_Tmax, "Tmax")
        parse_float(self.in_m_dry, "m_dry")
        parse_float(self.in_m_prop, "m_prop")
        cfg["enable_wind"] = bool(self.var_wind.get())
        parse_float(self.in_lat, "latitude_deg")

        parse_float(self.in_Cd_sub, "aero_Cd_subsonic")
        parse_float(self.in_Cd_peak, "aero_Cd_transonic_peak")
        parse_float(self.in_Cd_sup, "aero_Cd_supersonic")
        parse_float(self.in_mach_start, "aero_mach_transonic_start")
        parse_float(self.in_mach_end, "aero_mach_transonic_end")
        parse_float(self.in_alpha_slope, "aero_Cd_alpha_slope")
        return cfg

    def on_save_config(self) -> None:
        try:
            cfg = self._collect_config()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        cfg_path = Path(self.entry_cfg.get())
        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            messagebox.showinfo("Saved", f"Wrote {cfg_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def on_load_config(self) -> None:
        path = filedialog.askopenfilename(title="Open config.json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        def set_if(key: str, setter):
            if key in cfg:
                setter(str(cfg[key]))
        set_if("diameter_m", self.in_diameter_m.set)
        set_if("A", self.in_A_override.set)
        set_if("Isp", self.in_Isp.set)
        set_if("Tmax", self.in_Tmax.set)
        set_if("m_dry", self.in_m_dry.set)
        set_if("m_prop", self.in_m_prop.set)
        if "enable_wind" in cfg:
            self.var_wind.set(bool(cfg["enable_wind"]))
        set_if("latitude_deg", self.in_lat.set)
        set_if("aero_Cd_subsonic", self.in_Cd_sub.set)
        set_if("aero_Cd_transonic_peak", self.in_Cd_peak.set)
        set_if("aero_Cd_supersonic", self.in_Cd_sup.set)
        set_if("aero_mach_transonic_start", self.in_mach_start.set)
        set_if("aero_mach_transonic_end", self.in_mach_end.set)
        set_if("aero_Cd_alpha_slope", self.in_alpha_slope.set)
        self.entry_cfg.set(path)

    def on_run_sim(self) -> None:
        # Save config first
        try:
            self.on_save_config()
        except Exception:
            return

        exe = self.entry_exe.get()
        workdir = self.entry_workdir.get()
        cfg_path = self.entry_cfg.get()

        if not exe:
            messagebox.showerror("Missing exe", "Please specify simulator executable path.")
            return
        if not os.path.isfile(exe):
            messagebox.showerror("Bad exe", f"File not found: {exe}")
            return
        if not workdir:
            messagebox.showerror("Missing working dir", "Please specify working directory.")
            return
        if not os.path.isdir(workdir):
            messagebox.showerror("Bad working dir", f"Folder not found: {workdir}")
            return
        if not os.path.isfile(cfg_path):
            messagebox.showerror("Missing config", f"Config not found: {cfg_path}")
            return

        def run_and_plot():
            try:
                cmd = [exe, "--config", cfg_path]
                completed = subprocess.run(
                    cmd,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    shell=False,
                    check=False,
                )
                if completed.returncode != 0:
                    err = completed.stderr or completed.stdout
                    raise RuntimeError(f"Simulator failed (code {completed.returncode}):\n{err}")
                traj_csv = os.path.join(workdir, "trajectory.csv")
                if not os.path.isfile(traj_csv):
                    raise FileNotFoundError(f"trajectory.csv not found in {workdir}")
                self._plot_trajectory(traj_csv)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Run failed", str(e)))

        threading.Thread(target=run_and_plot, daemon=True).start()

    def _plot_trajectory(self, csv_path: str) -> None:
        try:
            # Read simple CSV: t,x,y,vx,vy,m,q,mach
            t_list = []
            x_list = []
            y_list = []
            vx_list = []
            vy_list = []
            mach_list = []
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 8:
                        continue
                    t, x, y, vx, vy, _m, _q, mach = parts[:8]
                    try:
                        t_list.append(float(t))
                        x_list.append(float(x))
                        y_list.append(float(y))
                        vx_list.append(float(vx))
                        vy_list.append(float(vy))
                        mach_list.append(float(mach))
                    except ValueError:
                        continue

            # Compute speed magnitude
            vmag = [abs((vx ** 2 + vy ** 2) ** 0.5) for vx, vy in zip(vx_list, vy_list)]

            self.ax_xy.clear()
            self.ax_vt.clear()
            self.ax_mach_t.clear()

            self.ax_xy.plot(x_list, y_list, label="trajectory")
            self.ax_xy.set_xlabel("x (m)")
            self.ax_xy.set_ylabel("y (m)")
            self.ax_xy.set_title("Ground track (x-y)")
            self.ax_xy.grid(True)

            self.ax_vt.plot(t_list, vmag, label="|v| (m/s)")
            self.ax_vt.set_xlabel("t (s)")
            self.ax_vt.set_ylabel("speed (m/s)")
            self.ax_vt.set_title("Speed vs time")
            self.ax_vt.grid(True)

            self.ax_mach_t.plot(t_list, mach_list, color="tab:red", label="Mach")
            self.ax_mach_t.set_xlabel("t (s)")
            self.ax_mach_t.set_ylabel("Mach")
            self.ax_mach_t.set_title("Mach vs time")
            self.ax_mach_t.grid(True)

            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot failed", str(e))


def main() -> None:
    # Prefer TkAgg backend
    if sys.platform.startswith("win"):
        os.environ.setdefault("MPLBACKEND", "TkAgg")
    app = RocketUI()
    app.mainloop()


if __name__ == "__main__":
    main()


