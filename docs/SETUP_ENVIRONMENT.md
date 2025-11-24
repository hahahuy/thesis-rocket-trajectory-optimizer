# Environment Setup Guide

This guide provides comprehensive instructions for setting up the complete development environment for the rocket trajectory optimization project, including Python/Conda, C++ libraries, Windows-specific setup, and CI/CD configuration.

## Table of Contents

1. [Python Environment](#python-environment)
2. [Core C++ Libraries](#core-c-libraries)
3. [CMake Configuration](#cmake-configuration)
4. [Windows-Specific Setup](#windows-specific-setup)
5. [CI/CD Pipeline Setup](#cicd-pipeline-setup)
6. [Building the Project](#building-the-project)
7. [Troubleshooting](#troubleshooting)

---

## Python Environment

### Conda Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate rocket-pinn

# Or create manually
conda create -n rocket-pinn python=3.10
conda activate rocket-pinn
pip install -r requirements.txt
```

### Python Dependencies

Required packages:
- `numpy >= 1.20`
- `scipy >= 1.7`
- `matplotlib >= 3.4`
- `h5py >= 3.0`
- `casadi >= 3.6`
- `torch >= 2.0`
- `hydra-core >= 1.1`
- `pytest >= 7.0`

Verify installation:
```bash
python scripts/check_requirements.py
```

---

## Core C++ Libraries

### System Status

**Verified Working Libraries:**
- ✅ **GCC 15.2.1** and **Clang 20.1.8** with C++17/C++20 support
- ✅ **Eigen 3.4.0** - Linear algebra library
- ✅ **LibTorch 2.3.1** - PyTorch C++ frontend for neural networks
- ✅ **HDF5 1.14.6** - High-performance data storage
- ✅ **nlohmann/json 3.12.0** - JSON configuration management
- ✅ **spdlog 1.15.3** - Fast logging library
- ✅ **GoogleTest 1.17.0** - Unit testing framework
- ✅ **LAPACK/BLAS** - Linear algebra backends

**Requires Manual Setup:**
- ⚠️ **CasADi 3.6.7** with **IPOPT 3.14.16** - Optimal control and nonlinear optimization

### Linux Setup (Arch/CachyOS)

#### Prerequisites

```bash
sudo pacman -S --needed base-devel cmake ninja gcc gcc-fortran clang git wget unzip pkg-config
```

#### Install Core Libraries

1. **Eigen** - Linear algebra
   ```bash
   sudo pacman -S eigen
   ```

2. **LibTorch** - Neural networks and autograd
   ```bash
   # Already downloaded to external/libtorch/
   # Version: 2.3.1 CPU
   ```

3. **HDF5** - Data serialization
   ```bash
   sudo pacman -S hdf5
   ```

4. **Additional Libraries**
   ```bash
   sudo pacman -S nlohmann-json spdlog gtest lapack blas openblas
   ```

#### CasADi and IPOPT Setup

**Option 1: Automated Build Script (Recommended)**
```bash
# Run the automated build script (20-40 minutes)
./scripts/build_casadi_ipopt.sh
```

**Option 2: Manual Build**

1. **Build IPOPT**
   ```bash
   cd external/Ipopt
   mkdir build && cd build
   ../configure --prefix=/usr/local \
                --enable-shared \
                --with-blas="-lopenblas" \
                --with-lapack="-llapack"
   make -j$(nproc) && sudo make install
   ```

2. **Build CasADi**
   ```bash
   cd external/casadi
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DWITH_IPOPT=ON \
            -DWITH_LAPACK=ON
   make -j$(nproc) && sudo make install
   ```

**Option 3: Docker Environment**
```bash
# Build complete environment with all dependencies
docker build -t rocket-optimizer .

# Run interactive container
docker run -it --rm -v $(pwd):/workspace rocket-optimizer
```

### Linux Setup (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get install build-essential cmake libeigen3-dev libboost-all-dev \
    libyaml-cpp-dev libspdlog-dev libfmt-dev libgtest-dev libgmock-dev \
    libhdf5-dev

# Install Python dependencies
pip install pytest pytest-cov black flake8 isort
```

---

## Windows-Specific Setup

### Already Installed (Windows)

Based on installation check, the following are typically already set up:

- ✅ **Python 3.12.7** (Anaconda)
- ✅ **CMake 4.1.0**
- ✅ **Python Dependencies**:
  - numpy, scipy, matplotlib, h5py
  - casadi 3.7.2 (includes IPOPT)
  - torch 2.5.1+cu121
  - hydra-core, omegaconf, pytest
- ✅ **IPOPT with HSL Solvers**: All HSL solvers (ma97, ma86, ma77, ma57, ma27) and MUMPS are available

### Remaining Setup: C++ Dependencies

To build the C++ components, you need:

1. **Visual Studio Build Tools** (C++ compiler)
2. **Eigen3** (linear algebra library)
3. **Google Test (GTest)** (testing framework)

### Option 1: Using vcpkg (Recommended)

vcpkg is Microsoft's C++ package manager and works well on Windows.

#### Step 1: Install vcpkg

```powershell
# Clone vcpkg
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Integrate with Visual Studio (optional but recommended)
.\vcpkg integrate install
```

#### Step 2: Install Dependencies

```powershell
# Install Eigen3 and GTest
.\vcpkg install eigen3 gtest

# Note the installation path (usually C:\vcpkg\installed\x64-windows)
```

#### Step 3: Configure CMake with vcpkg

```powershell
cd <project-root>
mkdir build
cd build

# Configure with vcpkg toolchain
# Note: Update the path to match your vcpkg installation location
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

# Build
cmake --build . --config Release
```

**Note**: If you get a warning about `CMAKE_TOOLCHAIN_FILE` not being used, it's okay - vcpkg may have already integrated with your system. The configuration should still succeed.

### Option 2: Manual Installation

#### Install Visual Studio Build Tools

1. Download **Visual Studio Build Tools** or **Visual Studio Community** from:
   https://visualstudio.microsoft.com/downloads/

2. During installation, select:
   - **Desktop development with C++** workload
   - **Windows 10/11 SDK**
   - **CMake tools for Windows** (optional but helpful)

#### Install Eigen3

1. Download Eigen from: https://eigen.tuxfamily.org/
2. Extract to a location like `C:\eigen3`
3. Configure CMake with:
   ```powershell
   cmake .. -DEigen3_DIR=C:\eigen3\share\eigen3\cmake
   ```

#### Install Google Test

1. Download from: https://github.com/google/googletest/releases
2. Build and install, or use vcpkg (easier)

### Option 3: Using Conda (Alternative)

If you prefer using conda:

```powershell
# Install Eigen and GTest via conda-forge
conda install -c conda-forge eigen gtest

# Then configure CMake to use conda packages
cmake .. -DCMAKE_PREFIX_PATH=$env:CONDA_PREFIX
```

### Verify Windows Installation

After installing dependencies, test the setup:

```powershell
# Test Python requirements
python scripts/check_requirements.py

# Test IPOPT solvers
python scripts/test_linear_solvers.py

# Try to configure CMake (should find Eigen3 and GTest)
cd build
cmake ..
```

### Windows Troubleshooting

- **CMake can't find Eigen3**: Use `-DEigen3_DIR` to point to Eigen3's cmake directory, or use vcpkg toolchain file
- **CMake can't find GTest**: Install via vcpkg: `.\vcpkg install gtest`, or set `-DGTest_DIR`
- **Visual Studio compiler not found**: Install Visual Studio Build Tools, or use Developer Command Prompt for VS
- **IPOPT/CasADi issues**: CasADi 3.7.2 is already installed via pip and includes IPOPT. All HSL solvers are available and working.

---

## CMake Configuration

The project uses a modular CMake configuration:

```cmake
# Find core libraries
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(GTest REQUIRED)

# LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
find_package(Torch REQUIRED)

# CasADi (when available)
find_package(casadi REQUIRED)
```

### Building the Project

**With All Libraries:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # Linux/macOS
# or
cmake --build . --config Release  # Windows
```

**Testing Individual Components:**
```bash
# Test basic setup (Eigen, HDF5, JSON, spdlog, GoogleTest)
mkdir build_basic && cd build_basic
cp ../test_setup_cmake.txt CMakeLists.txt
cmake .. && make && ./test_setup

# Test with LibTorch
mkdir build_torch && cd build_torch  
cp ../test_libtorch_cmake.txt CMakeLists.txt
cmake .. && make && ./test_libtorch

# Test complete setup
mkdir build_complete && cd build_complete
cp ../test_complete_cmake.txt CMakeLists.txt
cmake .. && make && ./test_complete_setup
```

---

## CI/CD Pipeline Setup

### Overview

The project uses GitHub Actions for continuous integration with the following stages:

1. **Build C++ module** with CMake and run GoogleTests
2. **Run Python tests** with pytest
3. **Lint code** (clang-format for C++, black for Python)
4. **Smoke demo** with mini dataset

### Pipeline Stages

#### Stage 1: C++ Build and Test
- Installs system dependencies (Eigen3, Boost, yaml-cpp, spdlog, fmt, GoogleTest)
- Configures CMake with C++20 standard
- Builds the project in Release mode
- Runs GoogleTests with verbose output
- Tests LibTorch integration

#### Stage 2: Python Tests
- Sets up Python 3.10 environment
- Installs Python dependencies (numpy, scipy, matplotlib, h5py, casadi, hydra-core, torch)
- Runs pytest with coverage reporting
- Uploads coverage to Codecov

#### Stage 3: Code Linting
- **C++**: Uses clang-format with Google style
- **Python**: Uses black, flake8, and isort
- Checks code formatting and style compliance

#### Stage 4: Smoke Demo
- Creates mini dataset with physics-based trajectory
- Runs dynamics demo (timeout: 30s)
- Runs PINN training demo (timeout: 60s)
- Generates visualizations and saves artifacts

#### Stage 5: Build Artifacts (on main branch)
- Creates release build
- Packages artifacts
- Uploads build artifacts

### Local Development

#### Prerequisites
```bash
# System dependencies (Linux)
sudo apt-get install build-essential cmake libeigen3-dev libboost-all-dev \
    libyaml-cpp-dev libspdlog-dev libfmt-dev libgtest-dev libgmock-dev

# Python dependencies
pip install pytest pytest-cov black flake8 isort
```

#### Quick Commands
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

### Configuration Files

- **`.github/workflows/ci.yml`**: Main GitHub Actions workflow configuration
- **`.clang-format`**: C++ code formatting rules (Google style)
- **`pyproject.toml`**: Python project configuration with dependencies, Black formatting, isort, mypy, pytest, coverage settings
- **`.flake8`**: Python linting configuration
- **`config.yaml`**: Hydra configuration for the application

### Test Structure

**Python Tests (`tests/test_python.py`):**
- **TestBasicFunctionality**: Tests core dependencies (numpy, torch, casadi, hydra)
- **TestPhysicsModels**: Tests physics calculations
- **TestOptimization**: Tests optimization algorithms
- **TestPerformance**: Performance tests (marked as slow)
- **TestIntegration**: Integration tests

**C++ Tests:**
- Uses GoogleTest framework
- Tests physics models and dynamics
- Validates numerical computations

---

## Building the Project

### Quick Start

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)  # Linux/macOS
# or
cmake --build . --config Release  # Windows

# Run tests
ctest -C Release
```

### Performance Test Results

The comprehensive test demonstrates:

1. **Rocket Dynamics Simulation**
   - Eigen-based physics integration
   - 100 time steps in <1ms
   - Trajectory optimization ready

2. **Neural Network Training**
   - LibTorch PINN implementation
   - Automatic differentiation working
   - GPU detection (CPU fallback)

3. **Data Pipeline**
   - HDF5 trajectory storage
   - JSON configuration management
   - Structured logging

---

## Troubleshooting

### LibTorch Issues
```bash
# Check LibTorch installation
ls external/libtorch/lib/
export LD_LIBRARY_PATH=$PWD/external/libtorch/lib:$LD_LIBRARY_PATH
```

### CasADi Build Issues
```bash
# Install additional dependencies
sudo pacman -S mumps-par scotch  # Arch/CachyOS
# or
sudo apt-get install libmumps-dev libscotch-dev  # Ubuntu/Debian

# Clean build
rm -rf external/casadi/build external/Ipopt/build
```

### HDF5 Linking Issues
```bash
# Check HDF5 installation
pkg-config --cflags --libs hdf5
```

### CMake Issues

**CMake not finding dependencies:**
```bash
# Install missing packages
sudo apt-get install libeigen3-dev libboost-all-dev  # Ubuntu/Debian
sudo pacman -S eigen boost  # Arch/CachyOS
```

**Python import errors:**
```bash
# Activate conda environment
conda activate rocket-pinn
pip install -e .
```

**LibTorch integration fails:**
```bash
# Check LibTorch installation
cd build
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

---

## Next Steps

1. **Immediate Development**: Use the working libraries (Eigen, LibTorch, HDF5, etc.) to start implementing your rocket trajectory optimization algorithms.

2. **CasADi Integration**: Run the build script when ready for optimal control methods.

3. **GPU Support**: Update LibTorch to CUDA version if GPU acceleration is needed.

4. **Deployment**: Use Docker for consistent deployment across environments.

The core libraries are ready for development! You can start implementing your PINN models, physics simulations, and optimization algorithms right away.

