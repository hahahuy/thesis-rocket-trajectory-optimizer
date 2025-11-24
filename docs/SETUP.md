# Setup Guide - Quick Start

This guide will help you set up the complete development environment for the Rocket Trajectory Optimizer project.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **C++ Compiler**: GCC 7+ or Clang 5+ (C++17/C++20 support)
- [ ] **CMake**: Version 3.15 or higher
- [ ] **Python**: 3.8+ (3.10+ recommended)
- [ ] **Package Manager**: 
  - Linux: `pacman` (Arch/CachyOS) or `apt-get` (Ubuntu/Debian)
  - Windows: `vcpkg` or Visual Studio
  - macOS: `homebrew`

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/hahahuy/thesis-rocket-trajectory-optimizer.git
   cd thesis-rocket-trajectory-optimizer
   ```

2. **Set up Python environment**
   - See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#python-environment) for detailed instructions

3. **Install C++ dependencies**
   - Linux: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#core-libraries)
   - Windows: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#windows-specific-setup)

4. **Set up HSL/MUMPS for IPOPT** (optional but recommended)
   - See [SETUP_HSL_MUMPS.md](SETUP_HSL_MUMPS.md) for instructions

5. **Build the project**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)  # or cmake --build . --config Release on Windows
   ```

6. **Run tests**
   ```bash
   ctest -C Release  # C++ tests
   pytest tests/ -v  # Python tests
   ```

## Setup Documentation

- **[SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md)**: Complete environment setup
  - Python/Conda environment
  - Core C++ libraries (Eigen, LibTorch, HDF5, etc.)
  - CMake configuration
  - Windows-specific instructions
  - CI/CD pipeline setup

- **[SETUP_HSL_MUMPS.md](SETUP_HSL_MUMPS.md)**: HSL/MUMPS linear solver setup
  - Quick start (use MUMPS - recommended)
  - Full HSL installation (for best performance)
  - Verification steps

## Platform-Specific Guides

- **Linux (Arch/CachyOS)**: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#linux-setup)
- **Windows**: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#windows-specific-setup)
- **macOS**: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#macos-setup) (if applicable)

## Verification

After setup, verify your installation:

```bash
# Test Python dependencies
python scripts/check_requirements.py

# Test linear solvers (IPOPT)
python scripts/test_linear_solvers.py

# Test C++ build
cd build && ctest -C Release
```

## Next Steps

Once setup is complete:

1. Read the [main README](../README.md) for project overview
2. Check [DESIGN.md](DESIGN.md) for architecture details
3. Review [wp1_comprehensive_description.md](wp1_comprehensive_description.md) for physics core
4. Review [wp2_comprehensive_description.md](wp2_comprehensive_description.md) for OCP solver

## Troubleshooting

Common issues and solutions:

- **CMake can't find dependencies**: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#troubleshooting)
- **HSL/MUMPS issues**: See [SETUP_HSL_MUMPS.md](SETUP_HSL_MUMPS.md#troubleshooting)
- **Python import errors**: See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md#python-environment)

For more detailed troubleshooting, refer to the specific setup guides above.

