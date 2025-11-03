#!/usr/bin/env python3
"""
Performance benchmark script for WP2 OCP solver.

Logs: IPOPT iterations, solve time, KKT residuals, mesh defect stats.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data_gen.solve_ocp import load_parameters, load_scales, solve_ocp
from src.solver.dynamics_casadi import compute_dynamics
import casadi as ca


def get_system_info():
    """Get system information for reproducibility."""
    import platform
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }
    
    # Get git hash
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        info['git_hash'] = result.stdout.strip() if result.returncode == 0 else 'unknown'
    except:
        info['git_hash'] = 'unknown'
    
    # Get CasADi version
    try:
        info['casadi_version'] = ca.__version__
    except:
        info['casadi_version'] = 'unknown'
    
    return info


def compute_defect_stats(X, U, t, params):
    """Compute defect statistics."""
    nx = X.shape[0]
    N = U.shape[1]
    dt = (t[-1] - t[0]) / N
    
    defects = []
    
    from src.solver.collocation import compute_hermite_simpson_step
    
    def f(x, u, p):
        return compute_dynamics(x, u, p)
    
    # Use symbolic variables for defect computation
    nx = X.shape[0]
    nu = U.shape[0]
    x_k_sym = ca.MX.sym('x_k', nx)
    x_kp1_sym = ca.MX.sym('x_kp1', nx)
    u_k_sym = ca.MX.sym('u_k', nu)
    u_kp1_sym = ca.MX.sym('u_kp1', nu)
    
    defect_sym = compute_hermite_simpson_step(
        f, x_k_sym, u_k_sym, x_kp1_sym, u_kp1_sym, dt, params
    )
    defect_func = ca.Function('defect', [x_k_sym, u_k_sym, x_kp1_sym, u_kp1_sym], [defect_sym])
    
    for k in range(N):
        x_k = X[:, k]
        x_kp1 = X[:, k + 1]
        u_k = U[:, k]
        u_kp1 = U[:, k] if k == N - 1 else U[:, k + 1]
        
        # Compute defect using function
        defect = np.array(defect_func(x_k, u_k, x_kp1, u_kp1)).flatten()
        
        defects.append(np.linalg.norm(defect))
    
    return {
        'mean': np.mean(defects),
        'max': np.max(defects),
        'min': np.min(defects),
        'std': np.std(defects),
        'values': defects
    }


def benchmark_solve(params, limits, ocp_config, scales, label="benchmark"):
    """Benchmark a single OCP solve."""
    print(f"\nBenchmarking: {label}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        result = solve_ocp(params, limits, ocp_config, scales)
        solve_time = time.time() - start_time
        
        stats = result['stats']
        
        if not stats['success']:
            print(f"  ✗ Solver did not converge")
            return None
        
        # Compute defect statistics
        defect_stats = compute_defect_stats(
            result['X'], result['U'], result['t'], params
        )
        
        benchmark = {
            'label': label,
            'success': True,
            'iterations': stats['iterations'],
            'solve_time': solve_time,
            'solver_time': stats.get('solver_time', solve_time),
            'objective': float(stats['objective']),
            'constraint_violation': float(stats.get('constraint_violation', 0.0)),
            'defect_stats': {
                'mean': float(defect_stats['mean']),
                'max': float(defect_stats['max']),
                'min': float(defect_stats['min']),
                'std': float(defect_stats['std'])
            },
            'mesh_size': ocp_config['problem']['transcription']['n_intervals'],
            'tf': float(result['t'][-1]),
            'n_states': result['X'].shape[0],
            'n_controls': result['U'].shape[0]
        }
        
        print(f"  ✓ Converged in {stats['iterations']} iterations")
        print(f"  Solve time: {solve_time:.3f} s")
        print(f"  Objective: {stats['objective']:.6f}")
        print(f"  Max defect: {defect_stats['max']:.6e}")
        print(f"  Mean defect: {defect_stats['mean']:.6e}")
        
        return benchmark
        
    except RuntimeError as e:
        if "HSL" in str(e) or "Library loading failure" in str(e):
            print(f"  ✗ Solver not available: {e}")
            return None
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """Run performance benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark WP2 OCP solver')
    parser.add_argument('--output', type=str, default='experiments/wp2_benchmark.json',
                       help='Output JSON file')
    parser.add_argument('--mesh-sizes', type=int, nargs='+', default=[10, 20, 30],
                       help='Mesh sizes to benchmark')
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark (single mesh size)')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("WP2 OCP Solver Performance Benchmark")
    print("=" * 70)
    
    # System info
    sys_info = get_system_info()
    print(f"\nSystem Info:")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Git hash: {sys_info['git_hash'][:8]}")
    print(f"  CasADi: {sys_info['casadi_version']}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    # Load configurations
    params, limits, ocp_config = load_parameters(
        str(project_root / 'configs' / 'phys.yaml'),
        str(project_root / 'configs' / 'limits.yaml'),
        str(project_root / 'configs' / 'ocp.yaml')
    )
    scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
    
    # Standard test case
    params['m0'] = 50.0
    limits['T_max'] = 4000.0
    limits['m_dry'] = 10.0
    ocp_config['initial']['m'] = 50.0
    ocp_config['initial']['z'] = 0.0
    ocp_config['initial']['vz'] = 0.0
    ocp_config['problem']['time']['tf_fixed'] = 30.0
    
    # Run benchmarks
    benchmarks = []
    mesh_sizes = [10] if args.quick else args.mesh_sizes
    
    for N in mesh_sizes:
        ocp_config['problem']['transcription']['n_intervals'] = N
        benchmark = benchmark_solve(params, limits, ocp_config, scales, f"N={N}")
        if benchmark:
            benchmarks.append(benchmark)
    
    # Save results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'system_info': sys_info,
        'timestamp': datetime.now().isoformat(),
        'benchmarks': benchmarks
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Benchmark results saved to: {output_path}")
    
    # Summary
    if benchmarks:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        for bm in benchmarks:
            print(f"  {bm['label']:10s}: {bm['iterations']:4d} iter, "
                  f"{bm['solve_time']:6.3f} s, defect_max={bm['defect_stats']['max']:.2e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
