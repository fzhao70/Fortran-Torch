# Fortran-Torch Installation Guide

Complete installation instructions for Fortran-Torch on all platforms.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Step-by-Step Installation](#step-by-step-installation)
  - [Step 1: Install System Dependencies](#step-1-install-system-dependencies)
  - [Step 2: Install LibTorch](#step-2-install-libtorch)
  - [Step 3: Build Fortran-Torch](#step-3-build-fortran-torch)
  - [Step 4: Verify Installation](#step-4-verify-installation)
- [Platform-Specific Guides](#platform-specific-guides)
  - [Ubuntu/Debian](#ubuntudebian)
  - [CentOS/RHEL/Rocky Linux](#centosrhelrocky-linux)
  - [macOS](#macos)
  - [Windows](#windows)
  - [HPC Systems](#hpc-systems)
- [Advanced Configuration](#advanced-configuration)
- [Docker Installation](#docker-installation)
- [Python Environment Setup](#python-environment-setup)
- [Integration with Existing Projects](#integration-with-existing-projects)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)
- [Getting Help](#getting-help)

---

## Quick Start

For experienced users on Linux:

```bash
# Install dependencies
sudo apt install cmake g++ gfortran git wget unzip

# Download LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# Build Fortran-Torch
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..
make -j$(nproc)
ctest
```

For detailed instructions, continue reading.

---

## Prerequisites

### Required Software

#### 1. CMake (>= 3.18)

CMake is the build system used by Fortran-Torch.

**Check if installed:**
```bash
cmake --version
```

**Install if needed:**
- Ubuntu/Debian: `sudo apt install cmake`
- CentOS/RHEL: `sudo yum install cmake` (may need EPEL)
- macOS: `brew install cmake`
- Windows: Download from https://cmake.org/download/

#### 2. C++ Compiler with C++17 Support

Required for building the LibTorch interface.

**Supported compilers:**
- GCC >= 7.0
- Clang >= 5.0
- Intel C++ >= 19.0
- MSVC >= 2019 (Windows)

**Check if installed:**
```bash
g++ --version      # GCC
clang++ --version  # Clang
```

**Install if needed:**
- Ubuntu/Debian: `sudo apt install g++`
- CentOS/RHEL: `sudo yum install gcc-c++`
- macOS: `xcode-select --install` or `brew install gcc`

#### 3. Fortran Compiler

Required for building the Fortran interface and examples.

**Supported compilers:**
- GNU Fortran (gfortran) >= 7.0 (recommended)
- Intel Fortran (ifort) >= 19.0
- NAG Fortran >= 7.0

**Check if installed:**
```bash
gfortran --version
```

**Install if needed:**
- Ubuntu/Debian: `sudo apt install gfortran`
- CentOS/RHEL: `sudo yum install gcc-gfortran`
- macOS: `brew install gcc`

#### 4. Python (>= 3.7)

Required for training and exporting PyTorch models.

**Check if installed:**
```bash
python3 --version
pip3 --version
```

**Install if needed:**
- Ubuntu/Debian: `sudo apt install python3 python3-pip`
- CentOS/RHEL: `sudo yum install python3 python3-pip`
- macOS: `brew install python`
- Windows: Download from https://www.python.org/downloads/

#### 5. Git

For cloning the repository.

**Check if installed:**
```bash
git --version
```

**Install if needed:**
- Ubuntu/Debian: `sudo apt install git`
- CentOS/RHEL: `sudo yum install git`
- macOS: Included with Xcode Command Line Tools
- Windows: Download from https://git-scm.com/

### Optional Software

#### CUDA Toolkit (for GPU Support)

Required only if you want GPU acceleration.

**Versions:** CUDA >= 11.0 recommended

**Check if installed:**
```bash
nvcc --version
nvidia-smi
```

**Install:**
- Download from https://developer.nvidia.com/cuda-downloads
- Follow NVIDIA's installation guide for your platform

#### MPI (for Distributed Computing)

Required only for parallel/distributed applications.

**Options:**
- OpenMPI
- MPICH
- Intel MPI

**Install:**
- Ubuntu/Debian: `sudo apt install libopenmpi-dev`
- CentOS/RHEL: `sudo yum install openmpi-devel`

---

## Step-by-Step Installation

### Step 1: Install System Dependencies

Choose instructions for your operating system:

#### Ubuntu 20.04/22.04/24.04

```bash
# Update package list
sudo apt update

# Install all required packages
sudo apt install -y \
    cmake \
    g++ \
    gfortran \
    git \
    wget \
    unzip \
    python3 \
    python3-pip

# Verify installations
cmake --version       # Should be >= 3.18
g++ --version         # Should be >= 7.0
gfortran --version    # Should be >= 7.0
python3 --version     # Should be >= 3.7
```

#### CentOS 8/9, RHEL 8/9, Rocky Linux 8/9

```bash
# Enable EPEL repository (if not already enabled)
sudo dnf install -y epel-release

# Install all required packages
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    cmake \
    gcc \
    gcc-c++ \
    gcc-gfortran \
    git \
    wget \
    unzip \
    python3 \
    python3-pip

# Verify installations
cmake --version
g++ --version
gfortran --version
python3 --version
```

**Note:** If CMake version is too old, install from source or use Snap:
```bash
sudo snap install cmake --classic
```

#### macOS (10.15+)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install cmake gcc git wget python

# Verify installations
cmake --version
g++-13 --version      # Homebrew GCC version may vary
gfortran-13 --version
python3 --version
```

**Note:** macOS Xcode clang doesn't include Fortran. Use Homebrew GCC.

#### Windows 10/11

**Option 1: Using MSYS2 (Recommended)**

1. Download and install MSYS2 from https://www.msys2.org/

2. Open MSYS2 MinGW 64-bit terminal

3. Update package database:
```bash
pacman -Syu
```

4. Close and reopen terminal, then install packages:
```bash
pacman -S --needed \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-gcc-fortran \
    mingw-w64-x86_64-python \
    git \
    unzip
```

**Option 2: Using Visual Studio**

1. Install Visual Studio 2019/2022 with C++ support
2. Install Intel Fortran Compiler (oneAPI)
3. Install CMake from https://cmake.org/download/
4. Install Python from https://www.python.org/downloads/
5. Install Git from https://git-scm.com/

### Step 2: Install LibTorch

LibTorch is the C++ API for PyTorch and must be installed separately.

#### Determine Which Version You Need

1. **CPU vs GPU:**
   - CPU-only: For systems without NVIDIA GPU or when GPU is not needed
   - CUDA: For GPU acceleration (requires NVIDIA GPU and CUDA Toolkit)

2. **Check CUDA version** (if using GPU):
```bash
nvcc --version
```

3. **ABI Version:**
   - Modern systems (GCC >= 5): Use cxx11-abi version (recommended)
   - Older systems (GCC < 5): Use pre-cxx11-abi version

#### Download LibTorch

**Linux CPU-only (cxx11-abi):**
```bash
cd $HOME
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

**Linux CUDA 11.8 (cxx11-abi):**
```bash
cd $HOME
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

**Linux CUDA 12.1 (cxx11-abi):**
```bash
cd $HOME
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

**macOS (CPU-only):**
```bash
cd $HOME
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip
unzip libtorch-macos-latest.zip
```

**Windows:**
1. Go to https://pytorch.org/get-started/locally/
2. Select:
   - PyTorch Build: **Stable**
   - Your OS: **Windows**
   - Package: **LibTorch**
   - Language: **C++/Java**
   - Compute Platform: **CPU** or your CUDA version
3. Download the ZIP file
4. Extract to `C:\libtorch` (or location of your choice)

#### Verify LibTorch Installation

```bash
# Linux/macOS
ls -la ~/libtorch/lib/
# Should show libtorch.so, libtorch_cpu.so, libc10.so, etc.

# Windows
dir C:\libtorch\lib
# Should show torch.lib, c10.lib, etc.
```

#### Set LibTorch Path (Optional but Recommended)

**Linux/macOS (.bashrc or .zshrc):**
```bash
export LIBTORCH_PATH=$HOME/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
```

**Windows (Environment Variables):**
1. Open System Properties → Advanced → Environment Variables
2. Add to PATH: `C:\libtorch\lib`

### Step 3: Build Fortran-Torch

#### Clone the Repository

```bash
cd $HOME
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch
```

#### Create Build Directory

```bash
mkdir build
cd build
```

#### Configure with CMake

**Basic configuration (Linux/macOS):**
```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
```

**With specific compilers:**
```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_Fortran_COMPILER=gfortran \
      ..
```

**With custom install location:**
```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch \
      -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch \
      ..
```

**Windows (MSYS2):**
```bash
cmake -G "MinGW Makefiles" \
      -DCMAKE_PREFIX_PATH=/c/libtorch \
      ..
```

**Windows (Visual Studio):**
```bash
cmake -G "Visual Studio 16 2019" \
      -DCMAKE_PREFIX_PATH=C:/libtorch \
      ..
```

#### Build

**Linux/macOS:**
```bash
make -j$(nproc)  # Use all CPU cores
# or specify number of cores
make -j4
```

**Windows (MSYS2):**
```bash
mingw32-make -j4
```

**Windows (Visual Studio):**
```bash
cmake --build . --config Release
```

#### Install (Optional)

**System-wide installation (Linux/macOS):**
```bash
sudo make install
```

**User installation:**
```bash
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
```

**Note:** System-wide installation allows easier integration with other projects.

### Step 4: Verify Installation

#### Run Basic Tests

```bash
# In the build directory
ctest --output-on-failure
```

**Expected output:**
```
Test project /path/to/Fortran-Torch/build
    Start 1: BasicTests
1/2 Test #1: BasicTests .......................   Passed    0.52 sec
    Start 2: InferenceTest
2/2 Test #2: InferenceTest ....................   Passed    1.23 sec

100% tests passed, 0 tests failed out of 2
```

#### Run Validation Script

```bash
cd ..  # Return to root directory
bash scripts/validate.sh
```

**Expected output should show:**
```
Passed:  39
Warnings: 4
Failed:  0
```

#### Test Python Integration

```bash
# Install PyTorch for Python
pip3 install torch numpy

# Create and export a test model
cd examples/python
python3 simple_model.py
```

**Expected output:**
```
Creating simple neural network...
Training model...
Epoch [10/100], Loss: ...
...
Model exported successfully!
```

#### Test Fortran Example

```bash
# Copy model to build directory
cp simple_model.pt ../../build/

# Run Fortran example
cd ../../build
./examples/fortran/simple_example
```

**Expected output:**
```
=========================================
Fortran-Torch Simple Example
=========================================

CUDA available:  F

Step 1: Loading TorchScript model...
Model loaded successfully!
...
Example completed successfully!
```

#### Create a Minimal Test Program

Create `test_install.f90`:

```fortran
program test_install
    use ftorch
    implicit none

    logical :: cuda_available

    print *, "==================================="
    print *, "Fortran-Torch Installation Test"
    print *, "==================================="
    print *, ""

    ! Test CUDA availability
    cuda_available = torch_cuda_available()
    if (cuda_available) then
        print *, "Status: CUDA is available"
    else
        print *, "Status: CPU-only mode"
    end if

    print *, ""
    print *, "Installation successful!"
    print *, "==================================="

end program test_install
```

**Compile and run:**

```bash
# Using build directory
gfortran test_install.f90 \
    -I./modules \
    -L. -lftorch -lfortran_torch_cpp \
    -Wl,-rpath,$HOME/libtorch/lib \
    -o test_install

./test_install
```

**If installed system-wide:**
```bash
gfortran test_install.f90 \
    -I/usr/local/include/fortran \
    -L/usr/local/lib -lftorch -lfortran_torch_cpp \
    -o test_install

./test_install
```

---

## Platform-Specific Guides

### Ubuntu/Debian

Complete installation from scratch:

```bash
#!/bin/bash
# Ubuntu/Debian Complete Installation Script

# 1. Update system
sudo apt update
sudo apt upgrade -y

# 2. Install dependencies
sudo apt install -y cmake g++ gfortran git wget unzip python3 python3-pip

# 3. Install PyTorch for Python
pip3 install torch numpy matplotlib

# 4. Download LibTorch
cd $HOME
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
rm libtorch-cxx11-abi-shared-with-deps-latest.zip

# 5. Set environment variables
echo 'export LIBTORCH_PATH=$HOME/libtorch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 6. Clone Fortran-Torch
cd $HOME
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch

# 7. Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j$(nproc)

# 8. Test
ctest --output-on-failure

# 9. Install (optional)
sudo make install

echo "Installation complete!"
```

### CentOS/RHEL/Rocky Linux

Complete installation from scratch:

```bash
#!/bin/bash
# CentOS/RHEL/Rocky Complete Installation Script

# 1. Enable EPEL
sudo dnf install -y epel-release

# 2. Install development tools
sudo dnf groupinstall -y "Development Tools"

# 3. Install dependencies
sudo dnf install -y cmake gcc gcc-c++ gcc-gfortran git wget unzip python3 python3-pip

# 4. Check CMake version (must be >= 3.18)
CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
if [ "$(printf '%s\n' "3.18" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.18" ]; then
    echo "CMake version too old, installing newer version..."
    sudo dnf install -y snapd
    sudo systemctl enable --now snapd.socket
    sudo ln -s /var/lib/snapd/snap /snap
    sudo snap install cmake --classic
fi

# 5. Install PyTorch for Python
pip3 install --user torch numpy matplotlib

# 6. Download LibTorch
cd $HOME
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
rm libtorch-cxx11-abi-shared-with-deps-latest.zip

# 7. Set environment variables
echo 'export LIBTORCH_PATH=$HOME/libtorch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 8. Clone Fortran-Torch
cd $HOME
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch

# 9. Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j$(nproc)

# 10. Test
ctest --output-on-failure

echo "Installation complete!"
```

### macOS

Complete installation from scratch:

```bash
#!/bin/bash
# macOS Complete Installation Script

# 1. Install Homebrew if needed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. Install dependencies
brew update
brew install cmake gcc git wget python

# 3. Install PyTorch for Python
pip3 install torch numpy matplotlib

# 4. Download LibTorch
cd $HOME
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip
unzip libtorch-macos-latest.zip
rm libtorch-macos-latest.zip

# 5. Set environment variables
echo 'export LIBTORCH_PATH=$HOME/libtorch' >> ~/.zshrc  # or ~/.bashrc
echo 'export DYLD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc

# 6. Clone Fortran-Torch
cd $HOME
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch

# 7. Build (use Homebrew GCC)
mkdir build && cd build
GCC_VERSION=$(ls /usr/local/bin/g++-* | tail -1 | grep -oE '[0-9]+$')
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch \
      -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-$GCC_VERSION \
      -DCMAKE_Fortran_COMPILER=/usr/local/bin/gfortran-$GCC_VERSION \
      ..
make -j$(sysctl -n hw.ncpu)

# 8. Test
ctest --output-on-failure

echo "Installation complete!"
```

**Important macOS Notes:**
- Use Homebrew GCC, not Apple Clang (no Fortran support)
- Use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`
- May need to allow apps from unidentified developers in Security settings

### Windows

#### Using MSYS2 (Recommended)

**Step-by-step guide:**

1. **Install MSYS2:**
   - Download from https://www.msys2.org/
   - Run installer, install to `C:\msys64` (default)

2. **Update MSYS2:**
   - Open "MSYS2 MinGW 64-bit" from Start Menu
   - Run: `pacman -Syu`
   - Close and reopen terminal
   - Run: `pacman -Su`

3. **Install packages:**
```bash
pacman -S --needed \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-gcc-fortran \
    mingw-w64-x86_64-python \
    mingw-w64-x86_64-python-pip \
    git \
    unzip \
    wget
```

4. **Install PyTorch:**
```bash
pip install torch numpy
```

5. **Download LibTorch:**
   - Go to https://pytorch.org/get-started/locally/
   - Select: Windows, LibTorch, C++/Java, CPU or CUDA
   - Download and extract to `C:\libtorch`

6. **Clone and build:**
```bash
cd /c/Users/$USER
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch
mkdir build && cd build
cmake -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH=/c/libtorch ..
mingw32-make -j4
```

7. **Add to PATH:**
   - Add `C:\libtorch\lib` to system PATH
   - Add `C:\msys64\mingw64\bin` to system PATH

8. **Test:**
```bash
ctest --output-on-failure
```

#### Using Visual Studio + Intel Fortran

1. **Install Visual Studio 2019/2022:**
   - Download Community Edition
   - Select "Desktop development with C++"

2. **Install Intel oneAPI:**
   - Download Intel oneAPI Base Toolkit
   - Download Intel oneAPI HPC Toolkit (for Fortran)
   - Install both

3. **Install CMake:**
   - Download from https://cmake.org/download/
   - Choose "Add to PATH" during installation

4. **Download LibTorch:**
   - Follow instructions above for Windows

5. **Build:**
   - Open "Intel oneAPI command prompt for Visual Studio"
```cmd
cd %USERPROFILE%
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -DCMAKE_PREFIX_PATH=C:/libtorch ..
cmake --build . --config Release
```

### HPC Systems

Most HPC systems use module systems. Here's a general guide:

#### Generic HPC Installation

```bash
#!/bin/bash
# HPC Installation Script

# 1. Load modules
module purge  # Start clean
module load cmake/3.20  # Or latest available
module load gcc/11.2    # Or latest available with Fortran
module load python/3.9  # Or latest available

# Optional: For GPU systems
# module load cuda/11.8

# 2. Check loaded modules
module list

# 3. Install PyTorch in user space
pip install --user torch numpy

# 4. Download LibTorch to scratch or home directory
cd $HOME  # or cd $SCRATCH
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
rm libtorch-cxx11-abi-shared-with-deps-latest.zip

# For CUDA systems:
# wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip

# 5. Set environment (add to ~/.bashrc or job scripts)
export LIBTORCH_PATH=$HOME/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH

# 6. Clone Fortran-Torch
git clone https://github.com/fzhao70/Fortran-Torch.git
cd Fortran-Torch

# 7. Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
make -j8  # Adjust based on allocation

# 8. Test
ctest --output-on-failure

echo "Installation complete!"
echo "Add these to your job scripts:"
echo "  module load cmake/3.20 gcc/11.2 python/3.9"
echo "  export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:\$LD_LIBRARY_PATH"
```

#### Example SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=fortran-torch-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# Load required modules
module load cmake/3.20
module load gcc/11.2
module load python/3.9

# Set library path
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH

# Run your program
cd $HOME/Fortran-Torch/build
./your_program
```

#### Common HPC Module Commands

```bash
# List available modules
module avail

# Search for specific module
module avail gcc
module avail cmake

# Load module
module load gcc/11.2

# Unload module
module unload gcc

# List loaded modules
module list

# Get module info
module show gcc/11.2
```

---

## Advanced Configuration

### Build Options

All CMake options can be set with `-D`:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DBUILD_EXAMPLES=ON \      # Build examples (default: ON)
      -DBUILD_TESTS=ON \          # Build tests (default: ON)
      -DCMAKE_BUILD_TYPE=Release \  # Build type: Release, Debug, RelWithDebInfo
      -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch \  # Install location
      -DCMAKE_VERBOSE_MAKEFILE=ON \  # Verbose build output
      ..
```

### Compiler Optimization Flags

**For production use (maximum performance):**
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native" \
      -DCMAKE_Fortran_FLAGS="-O3 -march=native" \
      ..
```

**For debugging:**
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-g -O0" \
      -DCMAKE_Fortran_FLAGS="-g -O0 -fcheck=all -fbacktrace" \
      ..
```

### Static vs Shared Libraries

By default, Fortran-Torch builds shared libraries. For static:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DBUILD_SHARED_LIBS=OFF \
      ..
```

### CUDA Configuration

**Specify CUDA architecture:**
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCMAKE_CUDA_ARCHITECTURES="70;75;80" \  # For V100, T4, A100
      ..
```

### Cross-Compilation

**Example for ARM64:**
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch-arm64 \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
      -DCMAKE_Fortran_COMPILER=aarch64-linux-gnu-gfortran \
      ..
```

### Multiple Versions

To maintain multiple versions:

```bash
# Version 1.0
cd Fortran-Torch-1.0
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch/1.0 ..
make install

# Version 2.0
cd Fortran-Torch-2.0
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/fortran-torch/2.0 ..
make install

# Switch between versions in your environment
export FORTRAN_TORCH_ROOT=/opt/fortran-torch/1.0  # or /2.0
```

---

## Docker Installation

### Using Pre-built Docker Image (Recommended)

```bash
# Pull image
docker pull fzhao70/fortran-torch:latest

# Run container
docker run -it --rm fzhao70/fortran-torch:latest
```

### Build Your Own Docker Image

Create `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    gfortran \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch numpy

# Download LibTorch
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-latest.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-latest.zip

# Set environment
ENV LIBTORCH_PATH=/opt/libtorch
ENV LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH

# Clone and build Fortran-Torch
WORKDIR /opt
RUN git clone https://github.com/fzhao70/Fortran-Torch.git && \
    cd Fortran-Torch && \
    mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH .. && \
    make -j$(nproc) && \
    make install

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
```

**Build and run:**
```bash
docker build -t fortran-torch:custom .
docker run -it --rm -v $(pwd):/workspace fortran-torch:custom
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  fortran-torch:
    image: fzhao70/fortran-torch:latest
    volumes:
      - ./workspace:/workspace
    working_dir: /workspace
    command: /bin/bash
```

**Run:**
```bash
docker-compose run fortran-torch
```

---

## Python Environment Setup

### Using Virtual Environments (Recommended)

**venv (standard library):**
```bash
# Create virtual environment
python3 -m venv ~/ftorch-env

# Activate
source ~/ftorch-env/bin/activate  # Linux/macOS
# or
~/ftorch-env\Scripts\activate  # Windows

# Install PyTorch
pip install torch numpy matplotlib

# Deactivate when done
deactivate
```

**To use in your workflow:**
```bash
# Always activate before training models
source ~/ftorch-env/bin/activate
python train_model.py
deactivate
```

### Using Conda

```bash
# Create environment
conda create -n ftorch python=3.10

# Activate
conda activate ftorch

# Install PyTorch
conda install pytorch cpuonly -c pytorch  # CPU
# or
conda install pytorch pytorch-cuda=11.8 -c pytorch  # CUDA 11.8

# Install additional packages
conda install numpy matplotlib

# Deactivate
conda deactivate
```

### System-wide Installation

```bash
# Install PyTorch system-wide (not recommended for development)
sudo pip3 install torch numpy matplotlib
```

---

## Integration with Existing Projects

### Using CMake

Create `CMakeLists.txt` in your project:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject Fortran CXX)

# Find PyTorch
set(CMAKE_PREFIX_PATH "/path/to/libtorch")
find_package(Torch REQUIRED)

# Find Fortran-Torch
set(FortranTorch_DIR "/path/to/fortran-torch/install")
find_package(FortranTorch REQUIRED)

# Your executable
add_executable(my_app
    src/main.f90
    src/module1.f90
)

# Link libraries
target_link_libraries(my_app
    ftorch
    fortran_torch_cpp
    ${TORCH_LIBRARIES}
)

# Include directories
target_include_directories(my_app PRIVATE
    ${FortranTorch_INCLUDE_DIRS}
)
```

### Using Makefiles

Example `Makefile`:

```makefile
# Paths
LIBTORCH = /path/to/libtorch
FTORCH_ROOT = /path/to/fortran-torch
FTORCH_BUILD = $(FTORCH_ROOT)/build

# Compiler flags
FC = gfortran
FFLAGS = -I$(FTORCH_BUILD)/modules -O2
LDFLAGS = -L$(FTORCH_BUILD) -L$(LIBTORCH)/lib
LIBS = -lftorch -lfortran_torch_cpp -ltorch -lc10 -ltorch_cpu
RPATH = -Wl,-rpath,$(LIBTORCH)/lib

# Your program
SRCS = main.f90 module1.f90
OBJS = $(SRCS:.f90=.o)
TARGET = my_app

# Build
$(TARGET): $(OBJS)
	$(FC) -o $@ $^ $(LDFLAGS) $(LIBS) $(RPATH)

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET) *.mod

.PHONY: clean
```

### Manual Compilation

```bash
# Compile Fortran source
gfortran -c my_program.f90 \
    -I/path/to/fortran-torch/build/modules

# Link
gfortran -o my_program my_program.o \
    -L/path/to/fortran-torch/build \
    -L/path/to/libtorch/lib \
    -lftorch -lfortran_torch_cpp \
    -ltorch -lc10 -ltorch_cpu \
    -Wl,-rpath,/path/to/libtorch/lib
```

### pkg-config Support

If Fortran-Torch is installed system-wide:

```bash
# Check if available
pkg-config --exists fortran-torch && echo "Found"

# Get compile flags
pkg-config --cflags fortran-torch

# Get link flags
pkg-config --libs fortran-torch

# Compile with pkg-config
gfortran my_program.f90 \
    $(pkg-config --cflags --libs fortran-torch) \
    -o my_program
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CMake Cannot Find Torch

**Error:**
```
CMake Error at CMakeLists.txt:12 (find_package):
  Could not find a package configuration file provided by "Torch"
```

**Solutions:**

a) Verify LibTorch path is correct:
```bash
ls -la /path/to/libtorch/share/cmake/Torch
```

b) Specify path explicitly:
```bash
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
```

c) Use environment variable:
```bash
export CMAKE_PREFIX_PATH=/path/to/libtorch
cmake ..
```

#### 2. Undefined References to Torch Symbols

**Error:**
```
undefined reference to `torch::jit::Module::Module()'
```

**Cause:** ABI mismatch between LibTorch and compiler

**Solutions:**

a) Check GCC version:
```bash
g++ --version
```

b) For GCC >= 5, use cxx11-abi LibTorch:
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
```

c) For GCC < 5, use pre-cxx11-abi:
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
```

d) Clean and rebuild:
```bash
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=/path/to/correct/libtorch ..
make
```

#### 3. Runtime Library Not Found

**Error:**
```
error while loading shared libraries: libtorch.so: cannot open shared object file
```

**Solutions:**

a) Add to LD_LIBRARY_PATH (Linux):
```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

b) Add to DYLD_LIBRARY_PATH (macOS):
```bash
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```

c) Make permanent (Linux):
```bash
echo 'export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

d) Use ldconfig (Linux, system-wide):
```bash
sudo sh -c 'echo "/path/to/libtorch/lib" > /etc/ld.so.conf.d/libtorch.conf'
sudo ldconfig
```

e) Set RPATH during build:
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_INSTALL_RPATH=/path/to/libtorch/lib \
      ..
```

#### 4. Fortran Module Incompatibility

**Error:**
```
Fatal Error: Can't open module file 'ftorch.mod' for reading
```
or
```
Fatal Error: File 'ftorch.mod' opened at (1) is not a GNU Fortran module file
```

**Cause:** Fortran .mod files are compiler-specific

**Solutions:**

a) Rebuild with same Fortran compiler:
```bash
# Check which compiler was used for build
cd build
grep Fortran_COMPILER CMakeCache.txt

# Rebuild if needed
rm -rf build
mkdir build && cd build
cmake -DCMAKE_Fortran_COMPILER=gfortran ..
make
```

b) Use same compiler for your project:
```bash
gfortran my_program.f90 ...  # Same as build
```

#### 5. CUDA Version Mismatch

**Error:**
```
CUDA error: no kernel image is available for execution on the device
```

**Cause:** LibTorch CUDA version doesn't match system CUDA

**Solutions:**

a) Check system CUDA version:
```bash
nvcc --version
nvidia-smi
```

b) Download matching LibTorch CUDA version:
```bash
# For CUDA 11.8
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip

# For CUDA 12.1
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip
```

c) Rebuild Fortran-Torch with correct LibTorch

#### 6. Model File Not Found

**Error:**
```
Error loading model, status: -2
PyTorch error: open file failed because of errno 2 on fopen
```

**Solutions:**

a) Check file exists:
```bash
ls -la model.pt
```

b) Use absolute path:
```fortran
model = torch_load_model('/full/path/to/model.pt')
```

c) Verify working directory:
```bash
pwd
```

d) Copy model to working directory:
```bash
cp /path/to/model.pt .
```

#### 7. Segmentation Fault

**Error:**
```
Segmentation fault (core dumped)
```

**Debugging steps:**

a) Run with debugger:
```bash
gdb ./your_program
(gdb) run
(gdb) backtrace  # After crash
```

b) Enable bounds checking (rebuild):
```bash
cmake -DCMAKE_Fortran_FLAGS="-g -fcheck=all -fbacktrace" ..
make
```

c) Check for null pointers:
```fortran
if (.not. c_associated(model%ptr)) then
    print *, "Model is null!"
    stop 1
end if
```

d) Verify array dimensions match:
```fortran
! If model expects (1, 10) but you provide (10)
real(real32) :: input(1, 10)  ! Correct
! not
real(real32) :: input(10)     # Wrong
```

#### 8. Python/PyTorch Version Mismatch

**Error:**
```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```
or
```
Could not load model with version X, current version is Y
```

**Solutions:**

a) Check PyTorch versions match:
```bash
python3 -c "import torch; print(torch.__version__)"
```

b) Retrain model with matching PyTorch version

c) Use compatible LibTorch version (same major.minor as PyTorch)

#### 9. Memory Issues

**Error:**
```
std::bad_alloc
```
or
```
CUDA out of memory
```

**Solutions:**

a) For CPU:
- Reduce batch size
- Use smaller model
- Process data in chunks

b) For GPU:
- Reduce batch size
- Use model.eval() mode
- Clear cache between inferences
- Use mixed precision

c) Monitor memory:
```bash
# CPU memory
free -h
top

# GPU memory
nvidia-smi
watch -n 1 nvidia-smi
```

#### 10. Permission Denied Errors

**Error:**
```
Permission denied
```

**Solutions:**

a) For installation:
```bash
# Use sudo for system-wide
sudo make install

# Or install to user directory
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
```

b) For file access:
```bash
# Check permissions
ls -la model.pt

# Fix if needed
chmod 644 model.pt
```

### Getting More Help

#### Enable Verbose Output

```bash
# CMake verbose
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..

# Make verbose
make VERBOSE=1

# Test verbose
ctest --verbose
ctest --output-on-failure
```

#### Check Build Logs

```bash
# CMake log
cat build/CMakeFiles/CMakeOutput.log
cat build/CMakeFiles/CMakeError.log

# Build log
make 2>&1 | tee build.log
```

#### System Information

Collect system information for bug reports:

```bash
#!/bin/bash
echo "=== System Information ==="
uname -a
echo ""

echo "=== Compiler Versions ==="
cmake --version
g++ --version
gfortran --version
python3 --version
echo ""

echo "=== LibTorch Info ==="
ls -la /path/to/libtorch/lib/libtorch.so
echo ""

echo "=== CUDA Info (if applicable) ==="
nvcc --version 2>/dev/null || echo "CUDA not installed"
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"
echo ""

echo "=== Environment ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
```

---

## Uninstallation

### Remove Installed Files

If installed with `make install`:

```bash
# From build directory
sudo make uninstall  # If available

# Manual removal (if installed to /usr/local)
sudo rm -f /usr/local/lib/libftorch.so*
sudo rm -f /usr/local/lib/libfortran_torch_cpp.so*
sudo rm -f /usr/local/include/fortran_torch.h
sudo rm -rf /usr/local/include/fortran
sudo rm -f /usr/local/bin/fortran-torch-*
```

### Remove Build Directory

```bash
cd Fortran-Torch
rm -rf build
```

### Remove LibTorch

```bash
rm -rf ~/libtorch
```

### Remove Python Packages

```bash
pip3 uninstall torch
```

### Remove Environment Variables

Edit `~/.bashrc` or `~/.zshrc` and remove:
```bash
export LIBTORCH_PATH=$HOME/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

### Complete Removal

```bash
# Remove all Fortran-Torch files
rm -rf ~/Fortran-Torch
rm -rf ~/libtorch

# Remove installed files
sudo rm -rf /usr/local/lib/libftorch*
sudo rm -rf /usr/local/lib/libfortran_torch*
sudo rm -f /usr/local/include/fortran_torch.h
sudo rm -rf /usr/local/include/fortran

# Remove Python packages
pip3 uninstall torch numpy

# Clean environment variables from ~/.bashrc
```

---

## Getting Help

### Documentation

- **README**: Complete API reference and usage examples
- **ARCHITECTURE.md**: Design patterns and internals
- **TESTING.md**: Testing guide
- **Weather Integration Guides**: Model-specific integration

### Online Resources

- **GitHub Repository**: https://github.com/fzhao70/Fortran-Torch
- **Issue Tracker**: https://github.com/fzhao70/Fortran-Torch/issues
- **Discussions**: https://github.com/fzhao70/Fortran-Torch/discussions

### Reporting Issues

When reporting issues, include:

1. **System information:**
   - OS and version
   - Compiler versions (g++, gfortran, cmake)
   - LibTorch version
   - CUDA version (if applicable)

2. **Error messages:**
   - Complete error output
   - Build logs
   - Runtime errors

3. **Steps to reproduce:**
   - Minimal code example
   - Build commands used
   - Expected vs actual behavior

4. **What you've tried:**
   - Troubleshooting steps already attempted
   - Any partial solutions

### Example Issue Template

```markdown
**System Information:**
- OS: Ubuntu 22.04
- Compiler: GCC 11.4.0, gfortran 11.4.0
- CMake: 3.22.1
- LibTorch: 2.1.0 (CPU, cxx11-abi)
- CUDA: N/A

**Description:**
Brief description of the issue

**Error Message:**
```
Full error output here
```

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Additional Context:**
Any other relevant information
```

---

## Next Steps

After successful installation:

1. **Run Examples:**
   ```bash
   cd examples/python
   python3 simple_model.py
   cd ../../build
   ./examples/fortran/simple_example
   ```

2. **Read Documentation:**
   - [API Reference](README.md#api-reference)
   - [Weather Model Integration](docs/WEATHER_MODEL_INTEGRATION.md)

3. **Create Your First Model:**
   - Train a PyTorch model
   - Export to TorchScript
   - Use in Fortran application

4. **Explore Advanced Features:**
   - GPU acceleration
   - Batch processing
   - Thread safety with OpenMP

5. **Join the Community:**
   - Star the repository
   - Report bugs
   - Contribute improvements

---

**Installation Complete! 🎉**

You're now ready to use PyTorch models in your Fortran applications!
