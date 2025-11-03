"""
Validate WP2 OCP results against WP1 reference trajectories.

This script:
1. Generates WP1 reference trajectory (if needed)
2. Solves WP2 OCP for same initial conditions
3. Compares trajectories and constraints
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import yaml
import h5py

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from python.data_gen.solve_ocp import load_parameters, load_scales, solve_ocp


def generate_wp1_reference(project_root: Path, output_path: str = "data/reference_case.csv"):
    """
    Generate WP1 reference trajectory using C++ validation executable.
    
    Args:
        project_root: Project root directory
        output_path: Output CSV file path
    """
    print("Generating WP1 reference trajectory...")
    
    # Build and run validate_dynamics if needed
    build_dir = project_root / 'build'
    exe_path = build_dir / 'validate_dynamics'
    
    if not exe_path.exists():
        print(f"Error: {exe_path} not found. Please build the project first.")
        print("  Run: cd build && cmake .. && make")
        return None
    
    # Run reference flight case
    try:
        result = subprocess.run(
            [str(exe_path), "--reference"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Warning: validate_dynamics returned {result.returncode}")
            print(result.stderr)
        
        # Check if CSV was created
        csv_path = project_root / output_path
        if csv_path.exists():
            print(f"WP1 reference saved to: {csv_path}")
            return csv_path
        else:
            print(f"Warning: Reference CSV not found at {csv_path}")
            return None
            
    except Exception as e:
        print(f"Error running validate_dynamics: {e}")
        return None


def load_wp1_reference(csv_path: Path) -> pd.DataFrame:
    """
    Load WP1 reference trajectory from CSV.
    
    Args:
        csv_path: Path to reference CSV file
        
    Returns:
        DataFrame with columns: t, altitude, q, n, velocity, mass, x, y, z, vx, vy, vz
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Reference file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def compare_trajectories(wp1_df: pd.DataFrame, wp2_result: dict, tolerance: float = 1e-3):
    """
    Compare WP1 and WP2 trajectories.
    
    Args:
        wp1_df: WP1 reference trajectory DataFrame
        wp2_result: WP2 OCP result dictionary
        tolerance: Tolerance for comparison
        
    Returns:
        comparison: Dictionary with comparison metrics
    """
    # Extract WP2 trajectory
    X_wp2 = wp2_result['X']
    U_wp2 = wp2_result['U']
    t_wp2 = wp2_result['t']
    
    # WP1 trajectory
    t_wp1 = wp1_df['t'].values
    alt_wp1 = wp1_df['altitude'].values
    v_mag_wp1 = wp1_df['velocity'].values
    m_wp1 = wp1_df['mass'].values
    q_wp1 = wp1_df['q'].values
    n_wp1 = wp1_df['n'].values
    
    # WP2 trajectory
    alt_wp2 = X_wp2[2, :]  # z-position (altitude)
    v_wp2 = X_wp2[3:6, :]
    v_mag_wp2 = np.linalg.norm(v_wp2, axis=0)
    m_wp2 = X_wp2[13, :]
    
    # Interpolate WP1 to WP2 time grid
    alt_wp1_interp = np.interp(t_wp2, t_wp1, alt_wp1)
    v_mag_wp1_interp = np.interp(t_wp2, t_wp1, v_mag_wp1)
    m_wp1_interp = np.interp(t_wp2, t_wp1, m_wp1)
    q_wp1_interp = np.interp(t_wp2, t_wp1, q_wp1)
    n_wp1_interp = np.interp(t_wp2, t_wp1, n_wp1)
    
    # Compute differences
    alt_diff = alt_wp2 - alt_wp1_interp
    v_diff = v_mag_wp2 - v_mag_wp1_interp
    m_diff = m_wp2 - m_wp1_interp
    
    # Statistics
    alt_rmse = np.sqrt(np.mean(alt_diff**2))
    v_rmse = np.sqrt(np.mean(v_diff**2))
    m_rmse = np.sqrt(np.mean(m_diff**2))
    
    alt_max_diff = np.max(np.abs(alt_diff))
    v_max_diff = np.max(np.abs(v_diff))
    m_max_diff = np.max(np.abs(m_diff))
    
    # Constraint comparison
    # Note: WP2 constraints should be satisfied by optimization
    q_wp1_max = np.max(q_wp1)
    n_wp1_max = np.max(n_wp1)
    
    comparison = {
        'altitude': {
            'rmse': alt_rmse,
            'max_diff': alt_max_diff,
            'wp1_final': alt_wp1[-1],
            'wp2_final': alt_wp2[-1]
        },
        'velocity': {
            'rmse': v_rmse,
            'max_diff': v_max_diff,
            'wp1_max': np.max(v_mag_wp1),
            'wp2_max': np.max(v_mag_wp2)
        },
        'mass': {
            'rmse': m_rmse,
            'max_diff': m_max_diff,
            'wp1_final': m_wp1[-1],
            'wp2_final': m_wp2[-1]
        },
        'constraints_wp1': {
            'q_max': q_wp1_max,
            'n_max': n_wp1_max
        },
        'passed': (
            alt_max_diff < tolerance * np.max(np.abs(alt_wp1)) and
            v_max_diff < tolerance * np.max(np.abs(v_mag_wp1)) and
            m_max_diff < tolerance * np.max(np.abs(m_wp1))
        )
    }
    
    return comparison, {
        't': t_wp2,
        'alt_wp1': alt_wp1_interp,
        'alt_wp2': alt_wp2,
        'v_wp1': v_mag_wp1_interp,
        'v_wp2': v_mag_wp2,
        'm_wp1': m_wp1_interp,
        'm_wp2': m_wp2,
        'q_wp1': q_wp1_interp,
        'n_wp1': n_wp1_interp
    }


def plot_comparison(comparison_data: dict, output_path: Path = None):
    """
    Plot comparison between WP1 and WP2 trajectories.
    
    Args:
        comparison_data: Dictionary with trajectory data
        output_path: Optional path to save figure
    """
    t = comparison_data['t']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Altitude
    axes[0, 0].plot(t, comparison_data['alt_wp1'] / 1e3, 'b--', label='WP1 (Reference)', linewidth=2)
    axes[0, 0].plot(t, comparison_data['alt_wp2'] / 1e3, 'r-', label='WP2 (OCP)', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Altitude [km]')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Altitude Comparison')
    
    # Velocity
    axes[0, 1].plot(t, comparison_data['v_wp1'], 'b--', label='WP1 (Reference)', linewidth=2)
    axes[0, 1].plot(t, comparison_data['v_wp2'], 'r-', label='WP2 (OCP)', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Velocity Comparison')
    
    # Mass
    axes[0, 2].plot(t, comparison_data['m_wp1'], 'b--', label='WP1 (Reference)', linewidth=2)
    axes[0, 2].plot(t, comparison_data['m_wp2'], 'r-', label='WP2 (OCP)', linewidth=2)
    axes[0, 2].set_xlabel('Time [s]')
    axes[0, 2].set_ylabel('Mass [kg]')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].set_title('Mass Comparison')
    
    # Differences
    alt_diff = comparison_data['alt_wp2'] - comparison_data['alt_wp1']
    v_diff = comparison_data['v_wp2'] - comparison_data['v_wp1']
    m_diff = comparison_data['m_wp2'] - comparison_data['m_wp1']
    
    axes[1, 0].plot(t, alt_diff, 'k-', linewidth=2)
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Altitude Difference [m]')
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Altitude Difference (WP2 - WP1)')
    
    axes[1, 1].plot(t, v_diff, 'k-', linewidth=2)
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Velocity Difference [m/s]')
    axes[1, 1].grid(True)
    axes[1, 1].set_title('Velocity Difference (WP2 - WP1)')
    
    # Constraints (WP1 only, as reference)
    axes[1, 2].plot(t, comparison_data['q_wp1'] / 1e3, 'b-', label='Dynamic Pressure', linewidth=2)
    axes[1, 2].plot(t, comparison_data['n_wp1'], 'r-', label='Load Factor', linewidth=2)
    axes[1, 2].set_xlabel('Time [s]')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_title('WP1 Constraints')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Comparison plot saved to: {output_path}")
    
    plt.show()


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate WP2 OCP against WP1 reference')
    parser.add_argument('--generate-wp1', action='store_true',
                       help='Generate WP1 reference trajectory')
    parser.add_argument('--wp1-csv', type=str, default='data/reference_case.csv',
                       help='Path to WP1 reference CSV')
    parser.add_argument('--tolerance', type=float, default=1e-2,
                       help='Tolerance for comparison (relative)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot comparison')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Save plot to file')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    
    # Generate WP1 reference if needed
    if args.generate_wp1:
        wp1_csv = generate_wp1_reference(project_root, args.wp1_csv)
        if wp1_csv is None:
            print("Failed to generate WP1 reference. Exiting.")
            return
    else:
        wp1_csv = project_root / args.wp1_csv
    
    # Load WP1 reference
    if not wp1_csv.exists():
        print(f"WP1 reference not found at {wp1_csv}")
        print("Run with --generate-wp1 to generate it.")
        return
    
    print(f"\nLoading WP1 reference from: {wp1_csv}")
    wp1_df = load_wp1_reference(wp1_csv)
    print(f"  Loaded {len(wp1_df)} time points")
    print(f"  Time range: {wp1_df['t'].min():.1f} - {wp1_df['t'].max():.1f} s")
    
    # Solve WP2 OCP (need to match WP1 initial conditions)
    print("\nSolving WP2 OCP...")
    
    # Load configurations
    params, limits, ocp_config = load_parameters(
        str(project_root / 'configs' / 'phys.yaml'),
        str(project_root / 'configs' / 'limits.yaml'),
        str(project_root / 'configs' / 'ocp.yaml')
    )
    scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
    
    # Match WP1 initial conditions (from validate_dynamics.cpp)
    # Initial state: m₀ = 50 kg, T = 4000 N
    ocp_config['initial']['m'] = 50.0
    ocp_config['initial']['z'] = 0.0
    ocp_config['initial']['vz'] = 0.0
    
    # Match physical parameters
    params['m0'] = 50.0
    limits['T_max'] = 4000.0
    limits['m_dry'] = 10.0
    limits['q_max'] = 40000.0
    limits['n_max'] = 5.0
    params['T_max'] = 4000.0
    params['m_dry'] = 10.0
    params['q_max'] = 40000.0
    params['n_max'] = 5.0
    
    # Solve
    wp2_result = solve_ocp(params, limits, ocp_config, scales)
    
    print(f"WP2 OCP solved:")
    print(f"  Objective: {wp2_result['stats']['objective']:.6f}")
    print(f"  Iterations: {wp2_result['stats']['iterations']}")
    print(f"  Success: {wp2_result['stats']['success']}")
    
    # Compare trajectories
    print("\nComparing trajectories...")
    comparison, comparison_data = compare_trajectories(wp1_df, wp2_result, args.tolerance)
    
    # Print results
    print("\n=== Validation Results ===")
    print(f"Altitude:")
    print(f"  RMSE: {comparison['altitude']['rmse']:.2f} m")
    print(f"  Max diff: {comparison['altitude']['max_diff']:.2f} m")
    print(f"  WP1 final: {comparison['altitude']['wp1_final']/1e3:.2f} km")
    print(f"  WP2 final: {comparison['altitude']['wp2_final']/1e3:.2f} km")
    
    print(f"\nVelocity:")
    print(f"  RMSE: {comparison['velocity']['rmse']:.2f} m/s")
    print(f"  Max diff: {comparison['velocity']['max_diff']:.2f} m/s")
    print(f"  WP1 max: {comparison['velocity']['wp1_max']:.2f} m/s")
    print(f"  WP2 max: {comparison['velocity']['wp2_max']:.2f} m/s")
    
    print(f"\nMass:")
    print(f"  RMSE: {comparison['mass']['rmse']:.2f} kg")
    print(f"  Max diff: {comparison['mass']['max_diff']:.2f} kg")
    print(f"  WP1 final: {comparison['mass']['wp1_final']:.2f} kg")
    print(f"  WP2 final: {comparison['mass']['wp2_final']:.2f} kg")
    
    print(f"\nWP1 Constraints:")
    print(f"  Max q: {comparison['constraints_wp1']['q_max']/1e3:.2f} kPa")
    print(f"  Max n: {comparison['constraints_wp1']['n_max']:.2f} g")
    
    if comparison['passed']:
        print("\n✓ Validation PASSED (within tolerance)")
    else:
        print("\n✗ Validation FAILED (exceeds tolerance)")
    
    # Plot if requested
    if args.plot or args.save_plot:
        plot_comparison(comparison_data, args.save_plot)


if __name__ == '__main__':
    main()

