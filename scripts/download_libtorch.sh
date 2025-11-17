#!/bin/bash

# Script to download and extract LibTorch
# Usage: ./download_libtorch.sh [cpu|cu118|cu121]

set -e

# Default to CPU version
COMPUTE_PLATFORM="${1:-cpu}"

# Determine OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=linux;;
    Darwin*)    OS_TYPE=macos;;
    *)          echo "Unsupported OS: ${OS}"; exit 1;;
esac

# Base URL
BASE_URL="https://download.pytorch.org/libtorch"

# Determine download URL
case "${COMPUTE_PLATFORM}" in
    cpu)
        if [ "${OS_TYPE}" = "linux" ]; then
            URL="${BASE_URL}/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip"
        elif [ "${OS_TYPE}" = "macos" ]; then
            URL="${BASE_URL}/cpu/libtorch-macos-latest.zip"
        fi
        ;;
    cu118)
        URL="${BASE_URL}/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip"
        ;;
    cu121)
        URL="${BASE_URL}/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip"
        ;;
    *)
        echo "Unknown compute platform: ${COMPUTE_PLATFORM}"
        echo "Usage: $0 [cpu|cu118|cu121]"
        exit 1
        ;;
esac

echo "========================================"
echo "LibTorch Download Script"
echo "========================================"
echo "OS: ${OS_TYPE}"
echo "Platform: ${COMPUTE_PLATFORM}"
echo "URL: ${URL}"
echo ""

# Check if libtorch already exists
if [ -d "libtorch" ]; then
    echo "Warning: libtorch directory already exists"
    read -p "Do you want to remove it and download again? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing libtorch directory..."
        rm -rf libtorch
    else
        echo "Keeping existing libtorch directory. Exiting."
        exit 0
    fi
fi

# Download
echo "Downloading LibTorch..."
if command -v wget &> /dev/null; then
    wget -O libtorch.zip "${URL}"
elif command -v curl &> /dev/null; then
    curl -L -o libtorch.zip "${URL}"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract
echo "Extracting..."
unzip -q libtorch.zip

# Cleanup
rm libtorch.zip

# Get absolute path
LIBTORCH_PATH="$(cd libtorch && pwd)"

echo ""
echo "========================================"
echo "LibTorch downloaded successfully!"
echo "========================================"
echo "Location: ${LIBTORCH_PATH}"
echo ""
echo "To build Fortran-Torch, run:"
echo "  mkdir build && cd build"
echo "  cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} .."
echo "  make -j\$(nproc)"
echo ""
