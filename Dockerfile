# Multi-stage Dockerfile for Rocket Trajectory Optimizer
FROM archlinux:latest AS base

# Update system and install basic packages
RUN pacman -Syu --noconfirm && \
    pacman -S --needed --noconfirm \
    base-devel \
    cmake \
    ninja \
    gcc \
    gcc-fortran \
    clang \
    git \
    wget \
    unzip \
    pkg-config \
    eigen \
    hdf5 \
    nlohmann-json \
    spdlog \
    gtest \
    lapack \
    blas \
    openblas

# Create working directory
WORKDIR /workspace

# Stage 2: Download and build external dependencies
FROM base AS dependencies

# Download LibTorch
RUN mkdir -p external && cd external && \
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.3.1+cpu.zip

# Clone CasADi and IPOPT (for future building)
RUN cd external && \
    git clone --depth 1 --branch 3.6.7 https://github.com/casadi/casadi.git && \
    git clone --depth 1 --branch releases/3.14.16 https://github.com/coin-or/Ipopt.git

# Build IPOPT (this takes a while)
RUN cd external/Ipopt && \
    mkdir build && cd build && \
    ../configure --prefix=/usr/local \
                 --enable-shared \
                 --with-blas="-lopenblas" \
                 --with-lapack="-llapack" && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Build CasADi with IPOPT
RUN cd external/casadi && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DWITH_IPOPT=ON \
        -DWITH_LAPACK=ON \
        -DINSTALL_INTERNAL_HEADERS=ON \
        -DWITH_BUILD_REQUIRED=ON && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Stage 3: Development environment
FROM dependencies AS development

# Copy project files
COPY . /workspace/

# Set environment variables
ENV CMAKE_PREFIX_PATH="/workspace/external/libtorch:/usr/local"
ENV LD_LIBRARY_PATH="/workspace/external/libtorch/lib:/usr/local/lib:$LD_LIBRARY_PATH"

# Create build directory
RUN mkdir -p build

# Build project
RUN cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="/workspace/external/libtorch:/usr/local" && \
    make -j$(nproc)

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

# Labels
LABEL org.opencontainers.image.title="Rocket Trajectory Optimizer"
LABEL org.opencontainers.image.description="Physics-Informed Neural Networks for Rocket Trajectory Optimization"
LABEL org.opencontainers.image.version="0.1.0"
