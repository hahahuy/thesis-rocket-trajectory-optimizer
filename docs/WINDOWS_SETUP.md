# Windows Setup Guide

This guide will help you set up the complete development environment for the Rocket Trajectory Optimizer on Windows.

## ✅ Already Installed

Based on the installation check, the following are already set up:

- ✅ **Python 3.12.7** (Anaconda)
- ✅ **CMake 4.1.0**
- ✅ **Python Dependencies**:
  - numpy 1.26.4
  - scipy 1.13.1
  - matplotlib 3.9.2
  - h5py 3.11.0
  - casadi 3.7.2 (includes IPOPT)
  - torch 2.5.1+cu121
  - hydra-core 1.3.2
  - omegaconf 2.3.0
  - pytest 7.4.4
- ✅ **IPOPT with HSL Solvers**: All HSL solvers (ma97, ma86, ma77, ma57, ma27) and MUMPS are available

## 🔧 Remaining Setup: C++ Dependencies

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
cd C:\Users\Hahuy\Documents\GitHubRepository\thesis-rocket-trajectory-optimizing
mkdir build
cd build

# Configure with vcpkg toolchain
# Note: Update the path to match your vcpkg installation location
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\Users\Hahuy\Documents\GitHubRepository\vcpkg\scripts\buildsystems\vcpkg.cmake

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

## 🧪 Verify Installation

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

## 📝 Quick Reference

### Python Environment

All Python dependencies are installed. You can verify with:
```powershell
python scripts/check_requirements.py
```

### C++ Build Commands

Once dependencies are installed:

```powershell
# Configure
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

# Build
cmake --build . --config Release

# Run tests
ctest -C Release
```

### Running Python Components

```powershell
# Test solver
python scripts/test_linear_solvers.py

# Run validation
python scripts/validate_wp2_full.py
```

## 🐛 Troubleshooting

### CMake can't find Eigen3

- Make sure Eigen3 is installed and in your PATH
- Use `-DEigen3_DIR` to point to Eigen3's cmake directory
- Or use vcpkg toolchain file

### CMake can't find GTest

- Install via vcpkg: `.\vcpkg install gtest`
- Or set `-DGTest_DIR` to GTest's cmake directory

### Visual Studio compiler not found

- Install Visual Studio Build Tools
- Or use Developer Command Prompt for VS
- Or set `-G "MinGW Makefiles"` to use MinGW (if installed)

### IPOPT/CasADi issues

- CasADi 3.7.2 is already installed via pip and includes IPOPT
- All HSL solvers are available and working
- No additional setup needed for Python usage

## 📚 Additional Resources

- Project README: `README.md`
- Setup guide (Linux): `docs/setup_guide.md`
- WP2 documentation: `docs/wp2_comprehensive_description.md`

## ✨ Next Steps

1. Install C++ dependencies (Eigen3, GTest) using one of the methods above
2. Configure and build the C++ components
3. Run tests to verify everything works
4. Start developing!

---

**Status**: Python environment is fully set up ✅  
**Remaining**: C++ dependencies (Eigen3, GTest) need to be installed

