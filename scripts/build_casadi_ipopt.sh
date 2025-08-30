#!/bin/bash
set -e

echo "=== Building CasADi with IPOPT Support ==="
echo "This script will build IPOPT and CasADi from source."
echo "Estimated build time: 20-40 minutes depending on your system."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check for required tools
for tool in gcc gfortran cmake make git; do
    if ! command -v $tool &> /dev/null; then
        print_error "$tool is not installed. Please install it first."
        exit 1
    fi
done

# Create external directory if it doesn't exist
mkdir -p external
cd external

# Build IPOPT
print_status "Building IPOPT..."
if [ ! -d "Ipopt" ]; then
    print_error "IPOPT source not found. Please clone it first:"
    print_error "cd external && git clone --depth 1 --branch releases/3.14.16 https://github.com/coin-or/Ipopt.git"
    exit 1
fi

cd Ipopt

# Get ThirdParty dependencies for IPOPT
print_status "Getting IPOPT third-party dependencies..."
cd ThirdParty

# Get MUMPS
if [ ! -d "Mumps" ]; then
    print_status "Downloading MUMPS..."
    git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git Mumps
    cd Mumps
    ./get.Mumps
    cd ..
fi

# Get Metis  
if [ ! -d "Metis" ]; then
    print_status "Downloading Metis..."
    git clone https://github.com/coin-or-tools/ThirdParty-Metis.git Metis
    cd Metis
    ./get.Metis
    cd ..
fi

cd .. # Back to Ipopt root

# Configure and build IPOPT
print_status "Configuring IPOPT..."
mkdir -p build
cd build

../configure \
    --prefix=/usr/local \
    --enable-shared \
    --with-blas="-lopenblas" \
    --with-lapack="-llapack" \
    --enable-mumps

print_status "Building IPOPT (this may take 15-20 minutes)..."
make -j$(nproc)

print_status "Installing IPOPT..."
sudo make install
sudo ldconfig

cd ../../ # Back to external

# Build CasADi
print_status "Building CasADi..."
if [ ! -d "casadi" ]; then
    print_error "CasADi source not found. Please clone it first:"
    print_error "cd external && git clone --depth 1 --branch 3.6.7 https://github.com/casadi/casadi.git"
    exit 1
fi

cd casadi
mkdir -p build
cd build

print_status "Configuring CasADi..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_IPOPT=ON \
    -DWITH_LAPACK=ON \
    -DINSTALL_INTERNAL_HEADERS=ON \
    -DWITH_BUILD_REQUIRED=ON \
    -DWITH_MUMPS=ON

print_status "Building CasADi (this may take 10-15 minutes)..."
make -j$(nproc)

print_status "Installing CasADi..."
sudo make install
sudo ldconfig

cd ../../../ # Back to project root

print_status "Build completed successfully!"
print_status "You can now use CasADi and IPOPT in your C++ projects."
print_warning "Remember to add /usr/local/lib to your LD_LIBRARY_PATH if needed:"
print_warning "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
