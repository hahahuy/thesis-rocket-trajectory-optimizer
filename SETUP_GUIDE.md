# Rocket Trajectory Optimizer - Library Setup Guide

This guide provides comprehensive instructions for setting up all required libraries and toolchain for the rocket trajectory optimization project using Physics-Informed Neural Networks (PINNs).

## ✅ System Status

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

## 🚀 Quick Start

### 1. Test Current Setup

```bash
# Test all working libraries
cd build_test
./test_complete_setup
```

You should see successful tests for physics simulation, neural network training, and data handling.

### 2. Run Individual Tests

```bash
# Test basic libraries only
./test_setup

# Test LibTorch specifically
./test_libtorch
```

## 📋 Detailed Setup Instructions

### Prerequisites

Ensure you have a CachyOS/Arch Linux system with:
```bash
sudo pacman -S --needed base-devel cmake ninja gcc gcc-fortran clang git wget unzip pkg-config
```

### Core Libraries (Already Installed ✅)

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

### CasADi and IPOPT Setup

#### Option 1: Automated Build Script

```bash
# Run the automated build script (20-40 minutes)
./scripts/build_casadi_ipopt.sh
```

#### Option 2: Manual Build

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

#### Option 3: Docker Environment

```bash
# Build complete environment with all dependencies
docker build -t rocket-optimizer .

# Run interactive container
docker run -it --rm -v $(pwd):/workspace rocket-optimizer
```

## 🛠️ Building the Project

### With All Libraries

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Testing Individual Components

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

## 🔧 CMake Configuration

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

## 📊 Performance Test Results

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

## 🐳 Docker Development

For a completely reproducible environment:

```bash
# Build development container
docker build -t rocket-optimizer .

# Mount project and run
docker run -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  rocket-optimizer

# Inside container
cd build && make && ./test_complete_setup
```

## 🔍 Troubleshooting

### LibTorch Issues
```bash
# Check LibTorch installation
ls external/libtorch/lib/
export LD_LIBRARY_PATH=$PWD/external/libtorch/lib:$LD_LIBRARY_PATH
```

### CasADi Build Issues
```bash
# Install additional dependencies
sudo pacman -S mumps-par scotch

# Clean build
rm -rf external/casadi/build external/Ipopt/build
```

### HDF5 Linking Issues
```bash
# Check HDF5 installation
pkg-config --cflags --libs hdf5
```

## 📁 Project Structure

```
thesis-rocket-trajectory-optimizer/
├── CMakeLists.txt                 # Main build configuration
├── Dockerfile                     # Complete environment
├── SETUP_GUIDE.md                 # This guide
├── external/                      # External dependencies
│   ├── libtorch/                  # ✅ PyTorch C++
│   ├── casadi/                    # ⚠️ Optimal control
│   └── Ipopt/                     # ⚠️ Nonlinear optimization
├── scripts/
│   └── build_casadi_ipopt.sh      # Automated build
├── src/                           # Source code modules
├── tests/                         # Unit tests
└── build_*/                       # Build directories
```

## ✨ Next Steps

1. **Immediate Development**: Use the working libraries (Eigen, LibTorch, HDF5, etc.) to start implementing your rocket trajectory optimization algorithms.

2. **CasADi Integration**: Run the build script when ready for optimal control methods.

3. **GPU Support**: Update LibTorch to CUDA version if GPU acceleration is needed.

4. **Deployment**: Use Docker for consistent deployment across environments.

The core libraries are ready for development! You can start implementing your PINN models, physics simulations, and optimization algorithms right away.
