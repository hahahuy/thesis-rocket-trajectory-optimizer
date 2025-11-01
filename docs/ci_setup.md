# CI/CD Setup Guide

This document describes the automated testing, linting, and build pipeline for the thesis rocket trajectory optimizer project.

## Overview

The project uses GitHub Actions for continuous integration with the following stages:

1. **Build C++ module** with CMake and run GoogleTests
2. **Run Python tests** with pytest
3. **Lint code** (clang-format for C++, black for Python)
4. **Smoke demo** with mini dataset

## Pipeline Stages

### Stage 1: C++ Build and Test
- Installs system dependencies (Eigen3, Boost, yaml-cpp, spdlog, fmt, GoogleTest)
- Configures CMake with C++20 standard
- Builds the project in Release mode
- Runs GoogleTests with verbose output
- Tests LibTorch integration

### Stage 2: Python Tests
- Sets up Python 3.10 environment
- Installs Python dependencies (numpy, scipy, matplotlib, h5py, casadi, hydra-core, torch)
- Runs pytest with coverage reporting
- Uploads coverage to Codecov

### Stage 3: Code Linting
- **C++**: Uses clang-format with Google style
- **Python**: Uses black, flake8, and isort
- Checks code formatting and style compliance

### Stage 4: Smoke Demo
- Creates mini dataset with physics-based trajectory
- Runs dynamics demo (timeout: 30s)
- Runs PINN training demo (timeout: 60s)
- Generates visualizations and saves artifacts

### Stage 5: Build Artifacts (on main branch)
- Creates release build
- Packages artifacts
- Uploads build artifacts

## Local Development

### Prerequisites
```bash
# System dependencies
sudo apt-get install build-essential cmake libeigen3-dev libboost-all-dev \
    libyaml-cpp-dev libspdlog-dev libfmt-dev libgtest-dev libgmock-dev

# Python dependencies
pip install pytest pytest-cov black flake8 isort
```

### Quick Commands
```bash
# Build project
mkdir build && cd build && cmake .. && make -j$(nproc)

# Run tests
pytest tests/ -v

# Format code
black .
clang-format -i src/**/*.cpp src/**/*.hpp

# Run smoke demo
./scripts/train_pinn.sh
```

## Configuration Files

### `.github/workflows/ci.yml`
Main GitHub Actions workflow configuration.

### `.clang-format`
C++ code formatting rules (Google style).

### `pyproject.toml`
Python project configuration with:
- Dependencies and metadata
- Black formatting configuration
- isort import sorting
- mypy type checking
- pytest testing configuration
- Coverage settings

### `.flake8`
Python linting configuration.

### `config.yaml`
Hydra configuration for the application.

## Test Structure

### Python Tests (`tests/test_python.py`)
- **TestBasicFunctionality**: Tests core dependencies (numpy, torch, casadi, hydra)
- **TestPhysicsModels**: Tests physics calculations
- **TestOptimization**: Tests optimization algorithms
- **TestPerformance**: Performance tests (marked as slow)
- **TestIntegration**: Integration tests

### C++ Tests
- Uses GoogleTest framework
- Tests physics models and dynamics
- Validates numerical computations

## Smoke Demo

The smoke demo (`scripts/train_pinn.sh`) demonstrates:
1. **Dataset Generation**: Creates physics-based trajectory data
2. **PINN Training**: Simple Physics-Informed Neural Network
3. **Visualization**: Generates plots and saves results

## Monitoring and Alerts

- **Coverage**: Code coverage reports uploaded to Codecov
- **Build Status**: GitHub Actions status badges
- **Artifacts**: Build artifacts stored for 90 days

## Troubleshooting

### Common Issues

1. **CMake not finding dependencies**
   ```bash
   # Install missing packages
   sudo apt-get install libeigen3-dev libboost-all-dev
   ```

2. **Python import errors**
   ```bash
   # Activate conda environment
   conda activate Thesis-rocket
   pip install -e .
   ```

3. **LibTorch integration fails**
   ```bash
   # Check LibTorch installation
   cd cpp && mkdir build && cd build
   cmake .. -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
   ```

### Debug Commands
```bash
# Check system dependencies
pkg-config --modversion eigen3
pkg-config --modversion boost

# Check Python packages
python -c "import torch; print(torch.__version__)"
python -c "import casadi; print(casadi.__version__)"

# Run specific tests
pytest tests/test_python.py::TestBasicFunctionality -v
```

## Performance Considerations

- **Parallel Builds**: Uses `make -j$(nproc)` for parallel compilation
- **Caching**: GitHub Actions caches dependencies
- **Timeouts**: Demo runs have timeouts to prevent hanging
- **Resource Limits**: Each job runs on ubuntu-latest (2 cores, 7GB RAM)

## Security

- **Dependencies**: All dependencies are pinned to specific versions
- **Secrets**: No secrets required for current setup
- **Permissions**: Minimal required permissions for GitHub Actions

## Future Enhancements

1. **Docker Support**: Add Docker-based builds
2. **Multi-Platform**: Support for Windows and macOS
3. **Performance Testing**: Add benchmark tests
4. **Documentation**: Auto-generate API documentation
5. **Release Automation**: Automatic version bumping and releases
