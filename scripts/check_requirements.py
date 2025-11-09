#!/usr/bin/env python3
"""Check if environment has all required packages for WP3."""

import sys
import importlib

REQUIRED = {
    "numpy": ">=1.20.0",
    "scipy": ">=1.7.0",
    "matplotlib": ">=3.3.0",
    "h5py": ">=3.1.0",
    "casadi": ">=3.5.0",
    "yaml": "PyYAML",
    "torch": ">=1.9.0",
    "pytest": ">=6.0.0",
}

OPTIONAL = {
    "hydra_core": "hydra-core>=1.1.0",
    "omegaconf": "omegaconf>=2.1.0",
}

def check_package(name, import_name=None):
    """Check if a package is installed and return version."""
    if import_name is None:
        import_name = name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None

def main():
    print("=== WP3 Requirements Check ===\n")
    
    all_ok = True
    missing = []
    
    # Check required packages
    print("Required packages:")
    for pkg, req in REQUIRED.items():
        import_name = pkg
        if pkg == "yaml":
            import_name = "yaml"
        installed, version = check_package(pkg, import_name)
        if installed:
            print(f"  ✓ {pkg:15s} {version:15s} ({req})")
        else:
            print(f"  ✗ {pkg:15s} MISSING        ({req})")
            missing.append(pkg)
            all_ok = False
    
    print("\nOptional packages:")
    for pkg, req in OPTIONAL.items():
        import_name = pkg.replace("_", "-")
        installed, version = check_package(pkg, import_name)
        if installed:
            print(f"  ✓ {pkg:15s} {version:15s} ({req})")
        else:
            print(f"  ○ {pkg:15s} not installed ({req})")
    
    # NumPy version check
    try:
        import numpy as np
        np_version = np.__version__
        major = int(np_version.split('.')[0])
        if major >= 2:
            print(f"\n⚠ NumPy {np_version} detected (2.x)")
            print("  Code uses NumPy 2.0 compatible syntax (np.bytes_ instead of np.string_)")
        else:
            print(f"\n✓ NumPy {np_version} detected (< 2.0)")
    except ImportError:
        pass
    
    print("\n" + "="*50)
    if all_ok:
        print("✓ All required packages are installed!")
        return 0
    else:
        print("✗ Missing required packages:")
        print(f"  pip install {' '.join(missing)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

