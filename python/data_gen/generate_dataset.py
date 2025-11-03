"""
Generate dataset of optimal trajectories by solving OCP for multiple scenarios.

Varies physical parameters, initial conditions, and constraints to create
diverse training data for PINN.
"""

import sys
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
from datetime import datetime
import argparse
from tqdm import tqdm

# Add parent directory to path
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))

from python.data_gen.solve_ocp import load_parameters, load_scales, solve_ocp, save_results


def create_parameter_variations(base_params: Dict, base_limits: Dict) -> List[Tuple[Dict, Dict]]:
    """
    Create list of parameter variations.
    
    Uses Latin Hypercube Sampling (LHS) or grid sampling to vary:
    - Initial mass
    - Drag coefficient
    - Isp
    - Wind conditions
    - Limits (q_max, n_max)
    
    Args:
        base_params: Base physical parameters
        base_limits: Base operational limits
        
    Returns:
        variations: List of (params, limits) tuples
    """
    variations = []
    
    # Simple grid sampling (can be replaced with LHS)
    m0_values = np.linspace(3000.0, 8000.0, 3)  # Initial mass [kg]
    Cd_values = np.linspace(0.2, 0.4, 3)       # Drag coefficient
    Isp_values = np.linspace(250.0, 350.0, 3)  # Specific impulse [s]
    q_max_values = np.linspace(30000.0, 50000.0, 2)  # Max dynamic pressure [Pa]
    n_max_values = np.linspace(4.0, 6.0, 2)     # Max load factor [g]
    
    total_combinations = len(m0_values) * len(Cd_values) * len(Isp_values) * len(q_max_values) * len(n_max_values)
    
    print(f"Generating {total_combinations} parameter variations...")
    
    for m0 in m0_values:
        for Cd in Cd_values:
            for Isp in Isp_values:
                for q_max in q_max_values:
                    for n_max in n_max_values:
                        params = base_params.copy()
                        limits = base_limits.copy()
                        
                        # Vary parameters
                        params['m0'] = m0
                        params['Cd'] = Cd
                        params['Isp'] = Isp
                        limits['q_max'] = q_max
                        limits['n_max'] = n_max
                        
                        # Update params dict with limits for solver
                        params['q_max'] = q_max
                        params['n_max'] = n_max
                        params['m_dry'] = base_limits.get('m_dry', 1000.0)
                        params['T_max'] = base_limits.get('T_max', 1000000.0)
                        
                        variations.append((params, limits))
    
    return variations


def create_scenario_variations(base_config: Dict) -> List[Dict]:
    """
    Create scenario variations (initial conditions, objectives, etc.).
    
    Args:
        base_config: Base OCP configuration
        
    Returns:
        configs: List of OCP configurations
    """
    configs = []
    
    # Vary initial conditions
    z0_values = [0.0, 100.0, 500.0]  # Initial altitude [m]
    vz0_values = [0.0, 10.0, 50.0]   # Initial vertical velocity [m/s]
    
    # Vary objectives
    objective_types = ['fuel_minimization', 'time_minimization']
    
    for z0 in z0_values:
        for vz0 in vz0_values:
            for obj_type in objective_types:
                config = base_config.copy()
                config['initial']['z'] = z0
                config['initial']['vz'] = vz0
                config['problem']['objective']['type'] = obj_type
                configs.append(config)
    
    return configs


def generate_dataset(
    base_params: Dict,
    base_limits: Dict,
    base_ocp_config: Dict,
    scales: Dict,
    output_dir: str,
    max_scenarios: int = None,
    seed: int = 42
):
    """
    Generate dataset of optimal trajectories.
    
    Args:
        base_params: Base physical parameters
        base_limits: Base operational limits
        base_ocp_config: Base OCP configuration
        scales: Scaling factors
        output_dir: Output directory for HDF5 files
        max_scenarios: Maximum number of scenarios to generate (None = all)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Create parameter variations
    param_variations = create_parameter_variations(base_params, base_limits)
    
    # Create scenario variations
    scenario_variations = create_scenario_variations(base_ocp_config)
    
    # Combine variations
    all_scenarios = []
    for params, limits in param_variations:
        for config in scenario_variations:
            all_scenarios.append((params, limits, config))
    
    # Limit number of scenarios if specified
    if max_scenarios is not None:
        indices = np.random.choice(len(all_scenarios), min(max_scenarios, len(all_scenarios)), replace=False)
        all_scenarios = [all_scenarios[i] for i in indices]
    
    print(f"\nGenerating dataset with {len(all_scenarios)} scenarios...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata file
    metadata = {
        'n_scenarios': len(all_scenarios),
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'base_params': base_params,
        'base_limits': base_limits
    }
    
    # Solve each scenario
    successful = 0
    failed = 0
    
    for i, (params, limits, config) in enumerate(tqdm(all_scenarios, desc="Solving OCPs")):
        try:
            # Solve OCP
            result = solve_ocp(params, limits, config, scales)
            
            # Check if successful
            if result['stats']['success']:
                # Save result
                output_path = os.path.join(output_dir, f"run_{i+1:04d}.h5")
                save_results(result, output_path)
                
                # Save parameter info
                with h5py.File(output_path, 'a') as f:
                    # Save parameter variation
                    params_grp = f.create_group('parameters')
                    for key, value in params.items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            params_grp.attrs[key] = value
                        elif isinstance(value, np.ndarray):
                            params_grp.create_dataset(key, data=value)
                        elif isinstance(value, list):
                            params_grp.attrs[key] = np.array(value).tolist()
                    
                    limits_grp = f.create_group('limits')
                    for key, value in limits.items():
                        if isinstance(value, (int, float)):
                            limits_grp.attrs[key] = value
                
                successful += 1
            else:
                print(f"\nWarning: Scenario {i+1} did not converge")
                failed += 1
                
        except Exception as e:
            print(f"\nError in scenario {i+1}: {e}")
            failed += 1
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"\nDataset generation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate dataset of optimal trajectories')
    parser.add_argument('--phys', type=str, default='configs/phys.yaml',
                       help='Physical parameters file')
    parser.add_argument('--limits', type=str, default='configs/limits.yaml',
                       help='Limits file')
    parser.add_argument('--ocp', type=str, default='configs/ocp.yaml',
                       help='OCP configuration file')
    parser.add_argument('--scales', type=str, default='configs/scales.yaml',
                       help='Scales file')
    parser.add_argument('--output', type=str, default='data/raw/ocp_runs',
                       help='Output directory')
    parser.add_argument('--max-scenarios', type=int, default=None,
                       help='Maximum number of scenarios to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configurations
    base_path = Path(__file__).parent.parent.parent
    params, limits, ocp_config = load_parameters(
        str(base_path / args.phys),
        str(base_path / args.limits),
        str(base_path / args.ocp)
    )
    scales = load_scales(str(base_path / args.scales))
    
    # Generate dataset
    output_dir = base_path / args.output
    generate_dataset(
        params, limits, ocp_config, scales,
        str(output_dir),
        max_scenarios=args.max_scenarios,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

