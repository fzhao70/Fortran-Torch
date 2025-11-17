#!/bin/bash

# Quick build script for Fortran-Torch
# Usage: ./build.sh [path_to_libtorch]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get libtorch path
if [ $# -eq 1 ]; then
    LIBTORCH_PATH="$1"
elif [ -d "libtorch" ]; then
    LIBTORCH_PATH="$(cd libtorch && pwd)"
elif [ -d "../libtorch" ]; then
    LIBTORCH_PATH="$(cd ../libtorch && pwd)"
else
    echo_error "LibTorch not found!"
    echo "Please provide path to libtorch or run scripts/download_libtorch.sh first"
    echo "Usage: $0 [path_to_libtorch]"
    exit 1
fi

echo "========================================"
echo "Fortran-Torch Build Script"
echo "========================================"
echo_info "LibTorch path: ${LIBTORCH_PATH}"
echo ""

# Check if libtorch path is valid
if [ ! -f "${LIBTORCH_PATH}/share/cmake/Torch/TorchConfig.cmake" ]; then
    echo_error "Invalid LibTorch path: TorchConfig.cmake not found"
    exit 1
fi

# Check for required tools
echo_info "Checking for required tools..."

if ! command -v cmake &> /dev/null; then
    echo_error "CMake not found. Please install CMake >= 3.18"
    exit 1
fi

if ! command -v gfortran &> /dev/null && ! command -v ifort &> /dev/null; then
    echo_error "Fortran compiler not found. Please install gfortran or ifort"
    exit 1
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo_error "C++ compiler not found. Please install g++ or clang++"
    exit 1
fi

echo_info "All required tools found!"
echo ""

# Create build directory
if [ -d "build" ]; then
    echo_warn "build directory already exists"
    read -p "Remove and rebuild? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf build
        mkdir build
    fi
else
    mkdir build
fi

cd build

# Configure
echo_info "Configuring with CMake..."
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_PATH}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_EXAMPLES=ON \
      ..

if [ $? -ne 0 ]; then
    echo_error "CMake configuration failed!"
    exit 1
fi

echo ""

# Build
echo_info "Building..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j${NPROC}

if [ $? -ne 0 ]; then
    echo_error "Build failed!"
    exit 1
fi

echo ""
echo "========================================"
echo_info "Build completed successfully!"
echo "========================================"
echo ""
echo "Examples built:"
echo "  - ./examples/fortran/simple_example"
echo "  - ./examples/fortran/weather_model_example"
echo ""
echo "To run the simple example:"
echo "  1. Train a model: cd ../examples/python && python simple_model.py"
echo "  2. Run example:   cd ../../build && ./examples/fortran/simple_example"
echo ""
echo "To install system-wide:"
echo "  sudo make install"
echo ""
