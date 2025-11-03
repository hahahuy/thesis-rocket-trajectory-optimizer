# Quick Start: HSL Libraries for IPOPT

## Current Status

✅ **HSL solvers can be created** but fail at runtime (libhsl.so not found)
✅ **MUMPS works** - use this as the default!

## Solution: Use MUMPS

The easiest solution is to configure IPOPT to use MUMPS instead of HSL:

### Option 1: Set in config (Recommended)

Edit `configs/ocp.yaml`:
```yaml
solver:
  linear_solver: "mumps"  # Use MUMPS (works without HSL)
```

### Option 2: Install HSL Libraries

If you want to use HSL solvers (ma97, ma86, etc.), you need to install HSL:

**Quick install on Arch/CachyOS:**
```bash
# Try to find if HSL is available via AUR
yay -S coinhsl  # or search for coinhsl package
```

**Or build from source:**
1. Register at http://www.hsl.rl.ac.uk/ipopt (free for academic use)
2. Download `coinhsl-*.tar.gz`
3. Build and install (see `docs/install_hsl_libraries.md`)

**Then set library path:**
```bash
# Find where HSL was installed
find /usr -name "libhsl.so" 2>/dev/null

# Add to library path (if found in /usr/local/lib)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

## Quick Test

Test MUMPS:
```bash
python3 scripts/test_linear_solvers.py
```

Then run integration tests:
```bash
pytest tests/test_solver_integration.py -v
```

## Verification

After setting `linear_solver: "mumps"`, test it:
```bash
python3 -c "
from python.data_gen.solve_ocp import load_parameters, load_scales, solve_ocp
from pathlib import Path
# ... (small solve test)
"
```

## Summary

- **Use MUMPS** (already works): Set `linear_solver: "mumps"` in `configs/ocp.yaml`
- **Or install HSL** if you need the best performance
- Auto-detection will prefer MUMPS if HSL fails at runtime
