#!/usr/bin/env python3
"""
Full WP2 validation script with reproducibility logging.

Combines WP1 reference generation, OCP solve, and comparison.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data_gen.validate_wp2_vs_wp1 import (
    generate_wp1_reference, load_wp1_reference, compare_trajectories
)
from python.data_gen.solve_ocp import load_parameters, load_scales, solve_ocp
from src.utils.reproducibility import (
    get_reproducibility_metadata, print_reproducibility_info
)


def validate_wp2_full(
    tolerance=0.01,
    generate_wp1=True,
    output_json=None,
    project_root=None
):
    """
    Full WP2 validation pipeline with reproducibility logging.
    
    Args:
        tolerance: Relative tolerance for comparison (default: 0.01 = 1%)
        generate_wp1: Whether to generate WP1 reference
        output_json: Path to save validation results JSON (optional)
        project_root: Project root directory (optional)
    
    Returns:
        bool: True if validation passes
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)
    
    print("=" * 70)
    print("WP2 Full Validation Pipeline")
    print("=" * 70)
    
    # Reproducibility info
    repro_info = get_reproducibility_metadata(project_root)
    print_reproducibility_info(project_root)
    
    # Step 1: Generate WP1 reference
    wp1_csv = project_root / 'data' / 'reference_case.csv'
    
    if generate_wp1 or not wp1_csv.exists():
        print(f"\n[Step 1] Generating WP1 reference trajectory...")
        wp1_path = generate_wp1_reference(project_root, str(wp1_csv))
        if wp1_path is None:
            print("ERROR: Failed to generate WP1 reference")
            return False
    else:
        print(f"\n[Step 1] Using existing WP1 reference: {wp1_csv}")
        wp1_path = wp1_csv
    
    # Load WP1 reference
    wp1_df = load_wp1_reference(wp1_path)
    print(f"  Loaded {len(wp1_df)} time points")
    
    # Step 2: Solve WP2 OCP
    print(f"\n[Step 2] Solving WP2 OCP...")
    
    params, limits, ocp_config = load_parameters(
        str(project_root / 'configs' / 'phys.yaml'),
        str(project_root / 'configs' / 'limits.yaml'),
        str(project_root / 'configs' / 'ocp.yaml')
    )
    scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
    
    # Match WP1 initial conditions
    ocp_config['initial']['m'] = 50.0
    ocp_config['initial']['z'] = 0.0
    ocp_config['initial']['vz'] = 0.0
    ocp_config['problem']['transcription']['n_intervals'] = 15  # Match robustness sweep for consistency
    ocp_config['problem']['time']['tf_fixed'] = 30.0
    
    # Disable path constraints for initial validation (can enable after debugging)
    ocp_config['path_constraints']['dynamic_pressure']['enabled'] = False
    ocp_config['path_constraints']['load_factor']['enabled'] = False
    ocp_config['path_constraints']['mass']['enabled'] = False
    
    params['m0'] = 50.0
    limits['T_max'] = 4000.0
    limits['m_dry'] = 10.0
    params['T_max'] = 4000.0
    params['m_dry'] = 10.0
    
    try:
        wp2_result = solve_ocp(params, limits, ocp_config, scales)
    except RuntimeError as e:
        if "HSL" in str(e) or "Library loading failure" in str(e):
            print(f"ERROR: IPOPT solver not available: {e}")
            return False
        raise
    
    stats = wp2_result['stats']
    if not stats['success']:
        print(f"ERROR: WP2 OCP did not converge")
        return False
    
    print(f"  ✓ Converged in {stats['iterations']} iterations")
    print(f"  Objective: {stats['objective']:.6f}")
    
    # Step 3: Compare trajectories
    print(f"\n[Step 3] Comparing WP1 and WP2 trajectories...")
    comparison, comparison_data = compare_trajectories(wp1_df, wp2_result, tolerance)
    
    # Results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    
    print(f"\nAltitude:")
    print(f"  RMSE: {comparison['altitude']['rmse']:.2f} m")
    print(f"  Max diff: {comparison['altitude']['max_diff']:.2f} m")
    print(f"  WP1 final: {comparison['altitude']['wp1_final']/1e3:.2f} km")
    print(f"  WP2 final: {comparison['altitude']['wp2_final']/1e3:.2f} km")
    
    print(f"\nVelocity:")
    print(f"  RMSE: {comparison['velocity']['rmse']:.2f} m/s")
    print(f"  Max diff: {comparison['velocity']['max_diff']:.2f} m/s")
    
    print(f"\nMass:")
    print(f"  RMSE: {comparison['mass']['rmse']:.2f} kg")
    print(f"  Max diff: {comparison['mass']['max_diff']:.2f} kg")
    
    passed = comparison['passed']
    
    if passed:
        print("\n✓ Validation PASSED (within tolerance)")
    else:
        print("\n✗ Validation FAILED (exceeds tolerance)")
    
    # Save results
    if output_json:
        output_path = project_root / output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'reproducibility': repro_info,
            'tolerance': tolerance,
            'wp2_stats': {
                'success': stats['success'],
                'iterations': stats['iterations'],
                'objective': float(stats['objective']),
                'constraint_violation': float(stats.get('constraint_violation', 0.0))
            },
            'comparison': {
                'altitude': {
                    'rmse': float(comparison['altitude']['rmse']),
                    'max_diff': float(comparison['altitude']['max_diff']),
                    'wp1_final': float(comparison['altitude']['wp1_final']),
                    'wp2_final': float(comparison['altitude']['wp2_final'])
                },
                'velocity': {
                    'rmse': float(comparison['velocity']['rmse']),
                    'max_diff': float(comparison['velocity']['max_diff']),
                    'wp1_max': float(comparison['velocity']['wp1_max']),
                    'wp2_max': float(comparison['velocity']['wp2_max'])
                },
                'mass': {
                    'rmse': float(comparison['mass']['rmse']),
                    'max_diff': float(comparison['mass']['max_diff']),
                    'wp1_final': float(comparison['mass']['wp1_final']),
                    'wp2_final': float(comparison['mass']['wp2_final'])
                },
                'passed': passed
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    return passed


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Full WP2 validation')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Relative tolerance (default: 0.01 = 1%%)')
    parser.add_argument('--generate-wp1', action='store_true',
                       help='Generate WP1 reference')
    parser.add_argument('--output', type=str, default='experiments/wp2_validation.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    passed = validate_wp2_full(
        tolerance=args.tolerance,
        generate_wp1=args.generate_wp1,
        output_json=args.output
    )
    
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
