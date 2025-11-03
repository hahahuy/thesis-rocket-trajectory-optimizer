#!/usr/bin/env python3
"""
Test OCP solve without scaling to isolate AD issue.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data_gen.solve_ocp import load_parameters, solve_ocp, load_scales

project_root = Path('.')
params, limits, ocp_config = load_parameters(
    str(project_root / 'configs' / 'phys.yaml'),
    str(project_root / 'configs' / 'limits.yaml'),
    str(project_root / 'configs' / 'ocp.yaml')
)

# Minimal problem
ocp_config['problem']['transcription']['n_intervals'] = 5
ocp_config['problem']['time']['tf_fixed'] = 10.0
ocp_config['initial']['m'] = 50.0
ocp_config['path_constraints']['dynamic_pressure']['enabled'] = False
ocp_config['path_constraints']['load_factor']['enabled'] = False
ocp_config['path_constraints']['mass']['enabled'] = False

params['m0'] = 50.0
params['T_max'] = limits.get('T_max', 4000.0)

# Test WITHOUT scaling
print("Testing WITHOUT scaling...")
try:
    result = solve_ocp(params, limits, ocp_config, scales={})  # No scaling
    if result['stats']['success']:
        print(f"✓ SUCCESS: {result['stats']['iterations']} iterations")
    else:
        print(f"✗ Failed to converge")
except Exception as e:
    error_msg = str(e)
    if "Invalid number" in error_msg or "NaN" in error_msg:
        print(f"✗ Still has AD issue even without scaling")
    else:
        print(f"✗ Error: {error_msg[:100]}")

# Test WITH scaling (baseline)
print("\nTesting WITH scaling (baseline)...")
scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
try:
    result = solve_ocp(params, limits, ocp_config, scales=scales)
    if result['stats']['success']:
        print(f"✓ SUCCESS: {result['stats']['iterations']} iterations")
    else:
        print(f"✗ Failed to converge")
except Exception as e:
    error_msg = str(e)
    if "Invalid number" in error_msg or "NaN" in error_msg:
        print(f"✗ AD issue (expected)")
    else:
        print(f"✗ Error: {error_msg[:100]}")

