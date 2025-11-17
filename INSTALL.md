# Installation Guide

This guide provides detailed installation instructions for Fortran-Torch on various platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installing LibTorch](#installing-libtorch)
- [Building Fortran-Torch](#building-fortran-torch)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- **CMake** >= 3.18
- **C++ Compiler** with C++17 support
  - GCC >= 7.0
  - Clang >= 5.0
  - Intel C++ Compiler >= 19.0
  - MSVC >= 2019 (Windows)
- **Fortran Compiler**
  - GNU Fortran (gfortran) >= 7.0
  - Intel Fortran (ifort) >= 19.0
  - NAG Fortran >= 7.0
- **Python** >= 3.7 (for training models)

### Optional

- **CUDA Toolkit** >= 11.0 (for GPU support)
- **MPI** (for distributed computing)

### Checking Your System

```bash
# Check CMake version
cmake --version

# Check C++ compiler
g++ --version    # or clang++ --version

# Check Fortran compiler
gfortran --version    # or ifort --version

# Check Python
python --version
```

## Installing LibTorch

LibTorch is the C++ distribution of PyTorch. You need to download it separately.

### Linux / macOS

#### CPU-Only Version

```bash
# Download
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip

# Extract
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# Move to preferred location (optional)
sudo mv libtorch /opt/libtorch
```

#### CUDA Version (Linux only)

For CUDA 11.8:
```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

For CUDA 12.1:
```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

**Note**: Download the version matching your CUDA installation. Check with: `nvcc --version`

#### Pre-CXX11 ABI (for older systems)

If you're using an older compiler or distribution, you might need the pre-CXX11 ABI version:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
```

### Windows

1. Download from: https://pytorch.org/get-started/locally/
2. Select:
   - PyTorch Build: Stable
   - Your OS: Windows
   - Package: LibTorch
   - Language: C++/Java
   - Compute Platform: CPU or your CUDA version
3. Extract the ZIP file to a location like `C:\libtorch`

## Building Fortran-Torch

### Basic Build (Linux/macOS)

```bash
# Clone the repository
git clone https://github.com/yourusername/Fortran-Torch.git
cd Fortran-Torch

# Create build directory
mkdir build
cd build

# Configure
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build
make -j$(nproc)

# Test (optional)
ctest

# Install (optional)
sudo make install
```

### Build with Specific Compilers

```bash
# Specify compilers
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_CXX_COMPILER=g++-11 \
      -DCMAKE_Fortran_COMPILER=gfortran-11 \
      ..
```

### Build Options

```bash
# Disable examples
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DBUILD_EXAMPLES=OFF \
      ..

# Specify installation prefix
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch \
      ..

# Enable verbose output
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      ..

# Debug build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_BUILD_TYPE=Debug \
      ..
```

## Platform-Specific Instructions

### Ubuntu / Debian

```bash
# Install prerequisites
sudo apt update
sudo apt install -y cmake g++ gfortran git wget unzip

# Install Python and PyTorch
sudo apt install -y python3 python3-pip
pip3 install torch numpy

# Download and install LibTorch (see above)

# Build Fortran-Torch
git clone https://github.com/yourusername/Fortran-Torch.git
cd Fortran-Torch
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j$(nproc)
```

### CentOS / RHEL / Rocky Linux

```bash
# Install prerequisites
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake gcc-gfortran wget unzip

# You may need a newer CMake
wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
chmod +x cmake-3.25.0-linux-x86_64.sh
sudo ./cmake-3.25.0-linux-x86_64.sh --prefix=/usr/local --skip-license

# Install Python and PyTorch
sudo yum install -y python3 python3-pip
pip3 install torch numpy

# Continue with LibTorch download and build (see above)
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install cmake gcc wget

# Install Python and PyTorch
brew install python
pip3 install torch numpy

# Download LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip
unzip libtorch-macos-latest.zip

# Build Fortran-Torch
git clone https://github.com/yourusername/Fortran-Torch.git
cd Fortran-Torch
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j$(sysctl -n hw.ncpu)
```

### Windows (with MSYS2/MinGW)

1. Install MSYS2 from https://www.msys2.org/

2. Open MSYS2 MinGW 64-bit terminal:

```bash
# Update package database
pacman -Syu

# Install tools
pacman -S mingw-w64-x86_64-cmake \
          mingw-w64-x86_64-gcc \
          mingw-w64-x86_64-gcc-fortran \
          git

# Download and extract LibTorch (see above)

# Build
git clone https://github.com/yourusername/Fortran-Torch.git
cd Fortran-Torch
mkdir build && cd build
cmake -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
mingw32-make -j4
```

### HPC Systems (with Modules)

Many HPC systems use environment modules:

```bash
# Load required modules
module load cmake/3.20
module load gcc/11.2.0
module load python/3.9
module load cuda/11.8  # if needed

# Install PyTorch in user directory
pip install --user torch numpy

# Download LibTorch to your home/scratch directory
cd $HOME
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# Build Fortran-Torch
cd /path/to/Fortran-Torch
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j8
```

## Verifying Installation

### Test the Build

```bash
# In the build directory
cd build

# Run simple example
./examples/fortran/simple_example

# If installed system-wide, test installation
fortran-torch-config --version
```

### Create a Test Program

Create `test.f90`:

```fortran
program test_ftorch
    use ftorch
    implicit none

    logical :: cuda_avail

    print *, 'Fortran-Torch test'
    cuda_avail = torch_cuda_available()
    print *, 'CUDA available:', cuda_avail

end program test_ftorch
```

Compile and run:

```bash
# If using build directory
gfortran test.f90 -I../build/modules -L../build -lftorch -lfortran_torch_cpp \
    -Wl,-rpath,/path/to/libtorch/lib -o test

# If installed
gfortran test.f90 $(pkg-config --cflags --libs fortran-torch) -o test

./test
```

## Troubleshooting

### CMake cannot find Torch

**Error**: `Could not find package Torch`

**Solution**: Ensure `CMAKE_PREFIX_PATH` points to libtorch directory:

```bash
cmake -DCMAKE_PREFIX_PATH=/full/path/to/libtorch ..
```

### Undefined references to torch symbols

**Error**: `undefined reference to 'torch::...'`

**Cause**: ABI mismatch between LibTorch and your compiler

**Solution**:
- For GCC >= 5: Use cxx11-abi version
- For older GCC: Use pre-cxx11-abi version

### Runtime library not found

**Error**: `error while loading shared libraries: libtorch.so`

**Solution**: Add libtorch to library path:

```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

# Or add to .bashrc for permanent fix
echo 'export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Fortran module file incompatibility

**Error**: `can't open module file 'ftorch.mod'`

**Cause**: Fortran module files are compiler-specific

**Solution**: Rebuild with the same Fortran compiler you're using for your project

### CUDA version mismatch

**Error**: CUDA-related runtime errors

**Solution**: Ensure LibTorch CUDA version matches your system CUDA:

```bash
# Check system CUDA version
nvcc --version

# Download matching LibTorch version
```

### Segmentation fault on model load

**Possible causes**:
1. Model file doesn't exist or path is wrong
2. LibTorch version mismatch (model trained with different PyTorch version)
3. Memory corruption

**Solutions**:
1. Verify model file path
2. Use same PyTorch version for training and inference
3. Run with debugger: `gdb ./your_program`

## Next Steps

After successful installation:

1. Run the examples:
   ```bash
   cd examples/python
   python simple_model.py
   cd ../../build
   ./examples/fortran/simple_example
   ```

2. Read the [API documentation](README.md#api-reference)

3. Integrate into your project (see [README.md](README.md))

## Getting Help

- Check [GitHub Issues](https://github.com/yourusername/Fortran-Torch/issues)
- Read [Troubleshooting section](README.md#troubleshooting)
- Contact: your.email@example.com
