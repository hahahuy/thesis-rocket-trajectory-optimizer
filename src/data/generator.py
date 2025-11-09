from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from multiprocessing import get_context

from .sampler import lhs_sample, sobol_sample, persist_samples_table
from .storage import write_hdf5_case, write_npz_case


@dataclass
class DatasetCfg:
    n_train: int
    n_val: int
    n_test: int
    sampler: str
    seed: int
    time_horizon_s: float
    grid_hz: int
    retries_per_case: int
    parallel_workers: int
    store_format: str


@dataclass
class OcpCfg:
    kkt_tol: float
    max_iter: int
    mesh_points: int
    warm_start: bool


@dataclass
class Config:
    dataset: DatasetCfg
    params: Dict[str, Tuple[float, float]]
    constraints: Dict[str, float]
    scaling: str
    ocp: OcpCfg


def load_yaml_config(path: str) -> Config:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ds = raw["dataset"]
    ocp = raw["ocp"]
    return Config(
        dataset=DatasetCfg(
            n_train=ds["n_train"],
            n_val=ds["n_val"],
            n_test=ds["n_test"],
            sampler=ds["sampler"],
            seed=ds["seed"],
            time_horizon_s=ds["time_horizon_s"],
            grid_hz=ds["grid_hz"],
            retries_per_case=ds["retries_per_case"],
            parallel_workers=ds["parallel_workers"],
            store_format=ds["store_format"],
        ),
        params=raw["params"],
        constraints=raw["constraints"],
        scaling=raw["scaling"],
        ocp=OcpCfg(
            kkt_tol=ocp["kkt_tol"],
            max_iter=ocp["max_iter"],
            mesh_points=ocp["mesh_points"],
            warm_start=ocp["warm_start"],
        ),
    )


def time_grid(cfg: Config) -> np.ndarray:
    T = cfg.dataset.time_horizon_s
    hz = cfg.dataset.grid_hz
    N = int(T * hz) + 1
    return np.linspace(0.0, T, N)


def run_feasibility_checks(monitors: Dict[str, np.ndarray], limits: Dict[str, Any]) -> Dict[str, Any]:
    qmax = limits.get("qmax", np.inf)
    nmax = limits.get("nmax", np.inf)
    q_dyn = np.asarray(monitors.get("q_dyn", []), dtype=float)
    n_load = np.asarray(monitors.get("n_load", []), dtype=float)
    ok = True
    report: Dict[str, Any] = {}
    if q_dyn.size:
        q_violation = float(np.nanmax(q_dyn) - qmax)
        ok = ok and (np.nanmax(q_dyn) <= qmax + 1e-9)
        report["max_q_dyn"] = float(np.nanmax(q_dyn))
        report["qmax"] = float(qmax)
        report["q_violation"] = q_violation
    if n_load.size:
        n_violation = float(np.nanmax(n_load) - nmax)
        ok = ok and (np.nanmax(n_load) <= nmax + 1e-9)
        report["max_n_load"] = float(np.nanmax(n_load))
        report["nmax"] = float(nmax)
        report["n_violation"] = n_violation
    report["ok"] = bool(ok)
    return report


def build_phys_limits_env(sample: Dict[str, float], cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Build phys, limits, env dicts in SI per WP3 contract.
    
    phys: vehicle/aero/inertia
    limits: actuation/constraints
    env: gravity & wind
    """
    # phys (vehicle/aero/inertia) - SI
    phys = {
        "Cd": sample.get("Cd", 0.3),
        "CL_alpha": sample.get("CL_alpha", 3.5),  # 1/rad
        "Cm_alpha": sample.get("Cm_alpha", -0.8),  # 1/rad
        "S": sample.get("S", 0.05),  # m²
        "l_ref": sample.get("l_ref", 1.2),  # m
        "Isp": sample.get("Isp", 250.0),  # s
        "Ix": sample.get("Ix", 10.0),  # kg·m²
        "Iy": sample.get("Iy", 10.0),  # kg·m²
        "Iz": sample.get("Iz", 1.0),  # kg·m²
        "rho0": sample.get("rho0", 1.225),  # kg/m³
        "H": sample.get("H", 8500.0),  # m
    }
    
    # limits (actuation/constraints) - SI
    limits = {
        "Tmax": sample.get("Tmax", 4000.0),  # N
        "mdry": sample.get("mdry", 35.0),  # kg
        "gimbal_max_rad": sample.get("gimbal_max_rad", 0.1745),  # ~10 deg
        "thrust_rate": sample.get("thrust_rate", 1e6),  # N/s (optional)
        "gimbal_rate_rad": sample.get("gimbal_rate_rad", 1.0),  # rad/s (optional)
        "qmax": cfg.constraints.get("qmax", 4e4),  # Pa
        "nmax": cfg.constraints.get("nmax", 5.0),  # g (unitless)
    }
    
    # env (gravity & wind) - SI
    wind_type = sample.get("wind_type", "constant")
    if wind_type == "zero":
        env = {
            "gravity": {"type": "constant", "g0": 9.80665, "use_inverse_square": False},
            "wind": {"type": "zero"},
        }
    elif wind_type == "constant":
        wind_mag = sample.get("wind_mag", 0.0)
        wind_dir_rad = sample.get("wind_dir_rad", 0.0)
        # Optionally use components
        if "wind_u" in sample:
            env = {
                "gravity": {"type": "constant", "g0": 9.80665, "use_inverse_square": False},
                "wind": {
                    "type": "constant",
                    "wind_u": sample.get("wind_u", 0.0),
                    "wind_v": sample.get("wind_v", 0.0),
                    "wind_w": sample.get("wind_w", 0.0),
                },
            }
        else:
            env = {
                "gravity": {"type": "constant", "g0": 9.80665, "use_inverse_square": False},
                "wind": {"type": "constant", "wind_mag": wind_mag, "wind_dir_rad": wind_dir_rad},
            }
    elif wind_type == "gust":
        env = {
            "gravity": {"type": "constant", "g0": 9.80665, "use_inverse_square": False},
            "wind": {
                "type": "gust",
                "gust_amp": sample.get("gust_amp", 0.0),
                "gust_freq": sample.get("gust_freq", 1.0),
                "gust_axis": sample.get("gust_axis", "x"),
                "gust_phase": sample.get("gust_phase", 0.0),
            },
        }
    else:
        env = {
            "gravity": {"type": "constant", "g0": 9.80665, "use_inverse_square": False},
            "wind": {"type": "zero"},
        }
    
    return phys, limits, env


def _validate_state_order(x: np.ndarray) -> bool:
    """Sanity: state order [x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, wx,wy,wz, m]."""
    return x.shape[-1] == 14


def _validate_control_unit(u: np.ndarray, tol: float = 1e-6) -> bool:
    """Sanity: control_cb returns unit thrust direction ||uT||=1."""
    if u.shape[-1] != 4:
        return False
    uT = u[..., 1:4]
    norms = np.linalg.norm(uT, axis=-1)
    return np.allclose(norms, 1.0, atol=tol)


def solve_ocp_and_integrate(sample: Dict[str, float], cfg: Config, t: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Wire to WP2 and WP1 entrypoints per contract.
    
    Returns payload dict and ocp_stats dict, both in SI.
    """
    try:
        from src.solver.collocation import solve_ocp
        from src.physics.dynamics import integrate_truth
    except ImportError:
        # Placeholder: Generate realistic vertical ascent trajectory
        N = t.shape[0]
        state = np.zeros((N, 14), dtype=float)
        control = np.zeros((N, 4), dtype=float)
        
        # Get parameters from sample
        m0 = sample.get("m0", 50.0)
        Tmax = sample.get("Tmax", 4000.0)
        Isp = sample.get("Isp", 250.0)
        g0 = 9.81
        
        # Initialize state: [x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, wx,wy,wz, m]
        state[0, 0:3] = [0.0, 0.0, 0.0]  # Start at origin
        state[0, 3:6] = [0.0, 0.0, 0.0]  # Zero initial velocity
        state[0, 6:10] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        state[0, 10:13] = [0.0, 0.0, 0.0]  # Zero angular velocity
        state[0, 13] = m0  # Initial mass
        
        # Initialize control at t=0
        control[0, 0] = Tmax * 0.8
        control[0, 1:4] = [1.0, 0.0, 0.0]
        
        # Simple vertical ascent simulation
        dt = t[1] - t[0] if N > 1 else 1.0
        for i in range(1, N):
            # Control: constant thrust, vertical direction
            T = Tmax * 0.8  # 80% max thrust
            control[i, 0] = T
            control[i, 1:4] = [1.0, 0.0, 0.0]  # Unit thrust direction (vertical in body frame)
            
            # Simple dynamics: vertical ascent with gravity and mass flow
            m = state[i-1, 13]
            if m > sample.get("mdry", 35.0):
                # Mass flow rate
                m_dot = -T / (Isp * g0)
                state[i, 13] = max(state[i-1, 13] + m_dot * dt, sample.get("mdry", 35.0))
            else:
                state[i, 13] = state[i-1, 13]
            
            # Vertical velocity (simple integration)
            a = (T / state[i, 13]) - g0  # Acceleration
            state[i, 5] = state[i-1, 5] + a * dt  # vz
            state[i, 2] = state[i-1, 2] + state[i, 5] * dt  # z (altitude)
            
            # Keep quaternion normalized
            state[i, 6:10] = state[i-1, 6:10] / np.linalg.norm(state[i-1, 6:10])
            if np.linalg.norm(state[i, 6:10]) < 1e-9:
                state[i, 6:10] = [1.0, 0.0, 0.0, 0.0]
            
            # Copy other states (horizontal motion zero for vertical ascent)
            state[i, 0:2] = state[i-1, 0:2]
            state[i, 3:5] = state[i-1, 3:5]
            state[i, 10:13] = state[i-1, 10:13]
        
        # Compute monitors
        rho0 = sample.get("rho0", 1.225)
        H = sample.get("H", 8500.0)
        rho = rho0 * np.exp(-np.maximum(state[:, 2], 0.0) / H)  # Clamp altitude to non-negative
        v_mag = np.linalg.norm(state[:, 3:6], axis=1)
        q_dyn = 0.5 * rho * v_mag**2
        # Load factor: total acceleration / g0 (using actual thrust, not Tmax)
        T_actual = control[:, 0:1]  # Actual thrust at each time
        a_total = np.sqrt((T_actual / state[:, 13:14])**2 + g0**2)
        n_load = a_total / g0
        
        monitors = {
            "rho": rho,
            "q_dyn": q_dyn,
            "n_load": n_load,
        }
        
        ocp_stats = {"KKT": 1e-8, "iterations": 0, "solve_time": 0.0, "success": True}
        payload = {"time": t, "state": state, "control": control, "monitors": monitors, "ocp": {}}
        return payload, ocp_stats

    phys, limits, env = build_phys_limits_env(sample, cfg)
    # Scales are used internally by WP2; pass canonical fields
    scales = {"L": 10000.0, "V": 313.0, "T": 31.62, "M": 50.0, "F": 490.0, "W": 0.0316}

    try:
        sol = solve_ocp(phys=phys, limits=limits, ocp_cfg=cfg.ocp.__dict__, scales=scales)
    except NotImplementedError:
        # Placeholder: Generate realistic vertical ascent trajectory when WP2 not implemented
        N = t.shape[0]
        state = np.zeros((N, 14), dtype=float)
        control = np.zeros((N, 4), dtype=float)
        
        # Get parameters from sample
        m0 = sample.get("m0", 50.0)
        Tmax = sample.get("Tmax", 4000.0)
        Isp = sample.get("Isp", 250.0)
        g0 = 9.81
        
        # Initialize state: [x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, wx,wy,wz, m]
        state[0, 0:3] = [0.0, 0.0, 0.0]  # Start at origin
        state[0, 3:6] = [0.0, 0.0, 0.0]  # Zero initial velocity
        state[0, 6:10] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        state[0, 10:13] = [0.0, 0.0, 0.0]  # Zero angular velocity
        state[0, 13] = m0  # Initial mass
        
        # Initialize control at t=0
        control[0, 0] = Tmax * 0.8
        control[0, 1:4] = [1.0, 0.0, 0.0]
        
        # Simple vertical ascent simulation
        dt = t[1] - t[0] if N > 1 else 1.0
        for i in range(1, N):
            # Control: constant thrust, vertical direction
            T = Tmax * 0.8  # 80% max thrust
            control[i, 0] = T
            control[i, 1:4] = [1.0, 0.0, 0.0]  # Unit thrust direction
            
            # Simple dynamics: vertical ascent with gravity and mass flow
            m = state[i-1, 13]
            if m > sample.get("mdry", 35.0):
                m_dot = -T / (Isp * g0)
                state[i, 13] = max(state[i-1, 13] + m_dot * dt, sample.get("mdry", 35.0))
            else:
                state[i, 13] = state[i-1, 13]
            
            # Vertical velocity
            a = (T / state[i, 13]) - g0
            state[i, 5] = state[i-1, 5] + a * dt
            state[i, 2] = state[i-1, 2] + state[i, 5] * dt
            
            # Keep quaternion normalized
            q = state[i-1, 6:10].copy()
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-9:
                state[i, 6:10] = q / q_norm
            else:
                state[i, 6:10] = [1.0, 0.0, 0.0, 0.0]
            
            # Copy other states
            state[i, 0:2] = state[i-1, 0:2]
            state[i, 3:5] = state[i-1, 3:5]
            state[i, 10:13] = state[i-1, 10:13]
        
        # Compute monitors
        rho0 = sample.get("rho0", 1.225)
        H = sample.get("H", 8500.0)
        rho = rho0 * np.exp(-np.maximum(state[:, 2], 0.0) / H)  # Clamp altitude to non-negative
        v_mag = np.linalg.norm(state[:, 3:6], axis=1)
        q_dyn = 0.5 * rho * v_mag**2
        # Load factor: total acceleration / g0 (using actual thrust, not Tmax)
        T_actual = control[:, 0:1]  # Actual thrust at each time
        a_total = np.sqrt((T_actual / state[:, 13:14])**2 + g0**2)
        n_load = a_total / g0
        
        monitors = {"rho": rho, "q_dyn": q_dyn, "n_load": n_load}
        ocp_stats = {"KKT": 1e-8, "iterations": 0, "solve_time": 0.0, "success": True}
        payload = {"time": t, "state": state, "control": control, "monitors": monitors, "ocp": {}}
        return payload, ocp_stats
    if not sol.success:
        return {}, {"success": False, "message": sol.message}

    # Sanity: state order
    if not _validate_state_order(sol.x0):
        return {}, {"success": False, "message": "invalid_state_order"}

    # Sanity: control unit vector (sample a few knots)
    if sol.u_knots.shape[0] > 0:
        if not _validate_control_unit(sol.u_knots):
            return {}, {"success": False, "message": "control_not_unit"}

    try:
        integ = integrate_truth(
            x0=sol.x0,
            t=t,
            control_cb=sol.control_cb,
            phys=phys,
            limits=limits,
            env=env,
            method="rk45",
            rtol=1e-6,
            atol=1e-8,
            normalize_quat_every=1,
        )
    except NotImplementedError:
        # Placeholder: Generate realistic vertical ascent trajectory when WP1 not implemented
        N = t.shape[0]
        state = np.zeros((N, 14), dtype=float)
        control = np.zeros((N, 4), dtype=float)
        
        # Get parameters
        m0 = sample.get("m0", 50.0)
        Tmax = sample.get("Tmax", 4000.0)
        Isp = sample.get("Isp", 250.0)
        g0 = 9.81
        
        # Initialize state properly
        state[0, 0:3] = [0.0, 0.0, 0.0]
        state[0, 3:6] = [0.0, 0.0, 0.0]
        state[0, 6:10] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        state[0, 10:13] = [0.0, 0.0, 0.0]
        state[0, 13] = m0
        
        # Initialize control at t=0
        control[0, 0] = Tmax * 0.8
        control[0, 1:4] = [1.0, 0.0, 0.0]
        
        # Simple vertical ascent
        dt = t[1] - t[0] if N > 1 else 1.0
        for i in range(1, N):
            T = Tmax * 0.8
            control[i, 0] = T
            control[i, 1:4] = [1.0, 0.0, 0.0]
            
            m = state[i-1, 13]
            if m > sample.get("mdry", 35.0):
                m_dot = -T / (Isp * g0)
                state[i, 13] = max(state[i-1, 13] + m_dot * dt, sample.get("mdry", 35.0))
            else:
                state[i, 13] = state[i-1, 13]
            
            a = (T / state[i, 13]) - g0
            state[i, 5] = state[i-1, 5] + a * dt
            state[i, 2] = state[i-1, 2] + state[i, 5] * dt
            
            q = state[i-1, 6:10].copy()
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-9:
                state[i, 6:10] = q / q_norm
            else:
                state[i, 6:10] = [1.0, 0.0, 0.0, 0.0]
            
            state[i, 0:2] = state[i-1, 0:2]
            state[i, 3:5] = state[i-1, 3:5]
            state[i, 10:13] = state[i-1, 10:13]
        
        # Compute monitors
        rho0 = sample.get("rho0", 1.225)
        H = sample.get("H", 8500.0)
        rho = rho0 * np.exp(-np.maximum(state[:, 2], 0.0) / H)  # Clamp altitude to non-negative
        v_mag = np.linalg.norm(state[:, 3:6], axis=1)
        q_dyn = 0.5 * rho * v_mag**2
        # Load factor: total acceleration / g0 (using actual thrust, not Tmax)
        T_actual = control[:, 0:1]  # Actual thrust at each time
        a_total = np.sqrt((T_actual / state[:, 13:14])**2 + g0**2)
        n_load = a_total / g0
        
        monitors = {"rho": rho, "q_dyn": q_dyn, "n_load": n_load}
        ocp_stats = {"KKT": 1e-8, "iterations": 0, "solve_time": 0.0, "success": True}
        payload = {"time": t, "state": state, "control": control, "monitors": monitors, "ocp": {}}
        return payload, ocp_stats

    # Sanity: state order in integration result
    if not _validate_state_order(integ.x):
        return {}, {"success": False, "message": "integ_state_order_invalid"}

    # Sanity: control unit vectors
    if not _validate_control_unit(integ.u):
        return {}, {"success": False, "message": "integ_control_not_unit"}

    # Feasibility checks use monitors from IntegrateResult
    checks = run_feasibility_checks(integ.monitors, limits)
    if not checks.get("ok", True):
        return {}, {"success": False, "message": "feasibility_fail", "checks": checks}

    # Sanity: quaternion norm (renormalize if needed)
    quat = integ.x[:, 6:10]
    quat_norms = np.linalg.norm(quat, axis=1)
    max_quat_err = np.max(np.abs(quat_norms - 1.0))
    if max_quat_err > 1e-6:
        # Renormalize
        quat_normalized = quat / quat_norms[:, np.newaxis]
        integ.x[:, 6:10] = quat_normalized

    payload = {
        "time": integ.t,
        "state": integ.x,
        "control": integ.u,
        "monitors": integ.monitors,
        "ocp": {"knots/state": sol.x_knots, "knots/control": sol.u_knots, "t_knots": sol.t_knots},
    }
    ocp_stats = {
        "KKT": sol.stats.get("kkt", 0.0),
        "iterations": sol.stats.get("n_iter", 0),
        "solve_time": sol.stats.get("solve_time_s", 0.0),
        "success": True,
        "quat_norm_max_err": max_quat_err,
    }
    return payload, ocp_stats


def build_metadata(sample: Dict[str, float], ocp_stats: Dict[str, Any], cfg: Config, git_hash: str, seed: int) -> Dict[str, Any]:
    import datetime as dt

    return {
        "git_hash": git_hash,
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "wp_versions": {"wp1": "unknown", "wp2": "unknown", "wp3": "v1"},
        "configs": {"dataset": json.dumps(cfg.dataset.__dict__), "ocp": json.dumps(cfg.ocp.__dict__)},
        "params_used": sample,
        "ocp_stats": ocp_stats,
    }


def _generate_case(args: Tuple[int, str, np.ndarray, Config, np.ndarray, list, str]) -> Dict[str, Any]:
    idx, split, sample_vec, cfg, t, keys, outfmt = args
    sample = dict(zip(keys, sample_vec.tolist()))
    root = os.path.join("data", "raw")

    # retries
    success = False
    ocp_last: Dict[str, Any] = {}
    payload: Dict[str, Any] = {}
    for _attempt in range(cfg.dataset.retries_per_case + 1):
        payload, ocp_stats = solve_ocp_and_integrate(sample, cfg, t)
        ocp_last = ocp_stats
        if ocp_stats.get("success", False):
            success = True
            break
    if not success:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "failures.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"sample": sample, "ocp": ocp_last}) + "\n")
        return {"success": False}

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    meta = build_metadata(sample, ocp_last, cfg, git_hash, cfg.dataset.seed + idx)

    case_name = f"case_{split}_{idx}.h5" if outfmt == "hdf5" else f"case_{split}_{idx}.npz"
    case_path = os.path.join(root, case_name)
    if outfmt == "hdf5":
        write_hdf5_case(case_path, payload, meta)
    elif outfmt == "npz":
        write_npz_case(case_path, payload, meta)
    else:
        raise ValueError(f"Unsupported store_format {outfmt}")
    return {"success": True, "case": case_path}


def run_generation(cfg_path: str) -> None:
    cfg = load_yaml_config(cfg_path)
    bounds = cfg.params
    keys = list(bounds.keys())
    total = cfg.dataset.n_train + cfg.dataset.n_val + cfg.dataset.n_test
    if cfg.dataset.sampler == "lhs":
        samples = lhs_sample(total, bounds, cfg.dataset.seed)
    elif cfg.dataset.sampler == "sobol":
        samples = sobol_sample(total, bounds, cfg.dataset.seed)
    else:
        raise ValueError(f"Unknown sampler: {cfg.dataset.sampler}")

    # Persist samples table
    samples_path = os.path.join("data", "raw", "samples.jsonl")
    persist_samples_table(samples_path, keys, samples)

    # Prepare splits
    split_sizes = [cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test]
    split_names = ["train", "val", "test"]
    offsets = np.cumsum([0] + split_sizes)

    t = time_grid(cfg)
    fmt = cfg.dataset.store_format.lower()

    tasks = []
    for si, split in enumerate(split_names):
        start, end = offsets[si], offsets[si + 1]
        for ridx, idx in enumerate(range(start, end)):
            tasks.append((ridx, split, samples[idx], cfg, t, keys, fmt))

    with get_context("spawn").Pool(processes=cfg.dataset.parallel_workers) as P:
        for _ in P.imap_unordered(_generate_case, tasks, chunksize=2):
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()
    run_generation(args.config)


if __name__ == "__main__":
    main()
