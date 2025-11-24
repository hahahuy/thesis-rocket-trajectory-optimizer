# HSL/MUMPS Linear Solver Setup

This guide explains how to set up HSL libraries or MUMPS for IPOPT linear solvers in the optimal control problem (OCP) solver.

## Quick Start: Use MUMPS (Recommended)

**MUMPS works out of the box** and is the easiest solution. No additional installation needed!

### Configure IPOPT to Use MUMPS

Edit `configs/ocp.yaml`:
```yaml
solver:
  linear_solver: "mumps"  # Use MUMPS (works without HSL)
```

### Quick Test

Test MUMPS:
```bash
python3 scripts/test_linear_solvers.py
```

Then run integration tests:
```bash
pytest tests/test_solver_integration.py -v
```

---

## HSL Libraries Setup (Optional - For Best Performance)

HSL (Harwell Subroutine Library) solvers provide better performance than MUMPS, especially for large problems. If HSL is already installed on your system, you can use it.

### Current Status

**Good news!** HSL libraries are often already installed and working on your system!

The test script shows available solvers:
- ✓ All HSL solvers available: `ma97`, `ma86`, `ma77`, `ma57`, `ma27`
- ✓ MUMPS available as alternative
- ✓ PARDISO available

### Verify HSL Installation

To verify which solvers are available:
```bash
python3 scripts/test_linear_solvers.py
```

This will show you all available linear solvers and recommend the best one.

### Solver Selection

The code auto-detects the best available solver. It will use:

1. **ma97** (best, parallel) - if available
2. **ma86** - if ma97 not available
3. **ma77** - next choice
4. **ma57** - sequential but robust
5. **ma27** - basic fallback
6. **mumps** - alternative if HSL fails

### Configuration

You can override the auto-detection in `configs/ocp.yaml`:

```yaml
solver:
  linear_solver: "ma97"  # or "ma86", "mumps", etc.
  # or use "auto" to let it detect
```

---

## Installing HSL Libraries (If Not Available)

If HSL solvers are not available on your system, you can install them.

### Option 1: Package Manager (Arch/CachyOS)

```bash
# Try to find if HSL is available via AUR
yay -S coinhsl  # or search for coinhsl package
```

### Option 2: Build from Source

1. **Register for HSL** (free for academic use):
   - Go to http://www.hsl.rl.ac.uk/ipopt
   - Register and download `coinhsl-*.tar.gz`

2. **Build and Install**:
   ```bash
   # Extract the archive
   tar -xzf coinhsl-*.tar.gz
   cd coinhsl-*
   
   # Configure and build
   ./configure --prefix=/usr/local
   make -j$(nproc)
   sudo make install
   ```

3. **Set Library Path**:
   ```bash
   # Find where HSL was installed
   find /usr -name "libhsl.so" 2>/dev/null
   
   # Add to library path (if found in /usr/local/lib)
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   
   # Or add to /etc/ld.so.conf.d/hsl.conf
   echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/hsl.conf
   sudo ldconfig
   ```

### Option 3: Use Coin-HSL (Alternative)

Coin-HSL is an open-source alternative:

```bash
# Download from https://github.com/coin-or-tools/ThirdParty-HSL
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
cd ThirdParty-HSL
./get.ASL
./configure
make
sudo make install
```

---

## Verification

After setting up HSL or MUMPS, verify it works:

```bash
# Test linear solvers
python3 scripts/test_linear_solvers.py

# Run integration tests
pytest tests/test_solver_integration.py -v

# Test a specific solver
python3 -c "
from src.solver.collocation import solve_ocp
# ... (small solve test with your chosen solver)
"
```

---

## Troubleshooting

### HSL Solvers Can Be Created But Fail at Runtime

**Symptom**: IPOPT can create HSL solver objects, but fails with "libhsl.so not found" at runtime.

**Solution**: 
1. Find where HSL libraries are installed:
   ```bash
   find /usr -name "libhsl.so" 2>/dev/null
   ```

2. Add to library path:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```

3. Or configure system-wide:
   ```bash
   echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/hsl.conf
   sudo ldconfig
   ```

### MUMPS Not Available

**Symptom**: MUMPS solver fails to initialize.

**Solution**: MUMPS is typically included with IPOPT. If it's not available:
1. Rebuild IPOPT with MUMPS support
2. Or use HSL solvers instead

### Auto-Detection Not Working

**Symptom**: Code doesn't automatically select the best solver.

**Solution**: Manually specify in `configs/ocp.yaml`:
```yaml
solver:
  linear_solver: "mumps"  # or "ma97", "ma86", etc.
```

---

## Summary

- **Use MUMPS** (already works): Set `linear_solver: "mumps"` in `configs/ocp.yaml` - **Recommended for quick start**
- **Or use HSL** if you need the best performance and it's available
- Auto-detection will prefer HSL if available, fall back to MUMPS if HSL fails at runtime
- All integration tests should work with either MUMPS or HSL

You're all set! 🚀

