#!/usr/bin/env python3
"""
Test which linear solvers are available for IPOPT.

Run this script to check if HSL or MUMPS libraries are installed.
"""
import sys
import casadi as ca
import numpy as np


def test_linear_solver(solver_name):
    """Test if a linear solver is available."""
    try:
        # Just test if we can create the solver (don't solve, as that may segfault)
        x = ca.MX.sym('x', 1)
        nlp = {
            'x': x,
            'f': x**2
        }
        
        opts = {
            'ipopt.linear_solver': solver_name,
            'ipopt.print_level': 0,
        }
        
        # Just creating the solver is enough to test availability
        solver = ca.nlpsol(f'test_{solver_name}', 'ipopt', nlp, opts)
        
        # If we got here without exception, solver is available
        print(f"  ✓ {solver_name:10s}: Available")
        return True
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "hsl" in error_msg or "libhsl" in error_msg or "library loading" in error_msg:
            print(f"  ✗ {solver_name:10s}: Not available (HSL library not found)")
        elif "mumps" in error_msg:
            print(f"  ✗ {solver_name:10s}: Not available (MUMPS library not found)")
        elif "unknown" in error_msg or "invalid" in error_msg:
            print(f"  ✗ {solver_name:10s}: Not available (unknown solver)")
        else:
            print(f"  ✗ {solver_name:10s}: Error - {str(e)[:60]}")
        return False


def main():
    print("=" * 70)
    print("Testing Linear Solvers for IPOPT")
    print("=" * 70)
    
    # Test HSL solvers (best performance)
    print("\nHSL Solvers (high performance):")
    print("-" * 70)
    hsl_solvers = ['ma97', 'ma86', 'ma77', 'ma57', 'ma27']
    hsl_available = []
    for solver in hsl_solvers:
        if test_linear_solver(solver):
            hsl_available.append(solver)
    
    # Test alternative solvers
    print("\nAlternative Solvers:")
    print("-" * 70)
    alt_solvers = ['mumps', 'pardiso']
    alt_available = []
    for solver in alt_solvers:
        if test_linear_solver(solver):
            alt_available.append(solver)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if hsl_available:
        print(f"\n✓ HSL solvers available: {', '.join(hsl_available)}")
        print(f"  Recommended: Use '{hsl_available[0]}' in solve_ocp.py")
    elif alt_available:
        print(f"\n⚠ HSL not available, but alternatives work: {', '.join(alt_available)}")
        print(f"  Recommended: Use '{alt_available[0]}' in solve_ocp.py")
    else:
        print("\n✗ No linear solvers available!")
        print("\nTo install:")
        print("  1. MUMPS (easiest): sudo pacman -S mumps")
        print("  2. Coin-HSL: See docs/install_hsl_libraries.md")
    
    print("\n" + "=" * 70)
    
    return 0 if (hsl_available or alt_available) else 1


if __name__ == '__main__':
    sys.exit(main())

