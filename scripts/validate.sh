#!/bin/bash

# Validation script for Fortran-Torch
# Checks code structure, dependencies, and correctness

# set -e intentionally not used - we want to check all items

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

echo_pass() {
    echo -e "${GREEN}[✓]${NC} $1"
}

echo_fail() {
    echo -e "${RED}[✗]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

PASS=0
FAIL=0
WARN=0

check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        echo_pass "$description exists: $file"
        ((PASS++))
        return 0
    else
        echo_fail "$description missing: $file"
        ((FAIL++))
        return 1
    fi
}

check_dir() {
    local dir=$1
    local description=$2

    if [ -d "$dir" ]; then
        echo_pass "$description exists: $dir"
        ((PASS++))
        return 0
    else
        echo_fail "$description missing: $dir"
        ((FAIL++))
        return 1
    fi
}

check_command() {
    local cmd=$1
    local description=$2

    if command -v "$cmd" &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -n1 || echo "unknown")
        echo_pass "$description available: $version"
        ((PASS++))
        return 0
    else
        echo_warn "$description not found: $cmd"
        ((WARN++))
        return 1
    fi
}

echo ""
echo "======================================"
echo " Fortran-Torch Validation"
echo "======================================"
echo ""

# Check directory structure
echo_check "Validating directory structure..."
check_dir "src" "Source directory"
check_dir "src/cpp" "C++ source directory"
check_dir "src/fortran" "Fortran source directory"
check_dir "include" "Include directory"
check_dir "examples" "Examples directory"
check_dir "examples/python" "Python examples directory"
check_dir "examples/fortran" "Fortran examples directory"
check_dir "tests" "Tests directory"
check_dir "scripts" "Scripts directory"
echo ""

# Check core files
echo_check "Validating core files..."
check_file "CMakeLists.txt" "Root CMake file"
check_file "src/cpp/fortran_torch.cpp" "C++ implementation"
check_file "src/fortran/ftorch.f90" "Fortran module"
check_file "include/fortran_torch.h" "C header"
echo ""

# Check documentation
echo_check "Validating documentation..."
check_file "README.md" "README"
check_file "INSTALL.md" "Installation guide"
check_file "ARCHITECTURE.md" "Architecture document"
check_file "CONTRIBUTING.md" "Contributing guide"
check_file "CHANGELOG.md" "Changelog"
check_file "LICENSE" "License file"
echo ""

# Check examples
echo_check "Validating examples..."
check_file "examples/python/simple_model.py" "Simple Python example"
check_file "examples/python/weather_model.py" "Weather model example"
check_file "examples/fortran/simple_example.f90" "Simple Fortran example"
check_file "examples/fortran/weather_model_example.f90" "Weather Fortran example"
echo ""

# Check tests
echo_check "Validating tests..."
check_file "tests/CMakeLists.txt" "Tests CMake file"
check_file "tests/fortran/test_basic.f90" "Basic tests"
check_file "tests/fortran/test_inference.f90" "Inference tests"
check_file "tests/python/create_test_model.py" "Test model creator"
check_file "tests/README.md" "Test documentation"
echo ""

# Check scripts
echo_check "Validating scripts..."
check_file "scripts/download_libtorch.sh" "LibTorch download script"
check_file "scripts/build.sh" "Build script"
check_file "scripts/run_tests.sh" "Test runner script"
echo ""

# Check required tools
echo_check "Checking required build tools..."
check_command "cmake" "CMake"
check_command "make" "Make"
check_command "gfortran" "Fortran compiler (gfortran)" || check_command "ifort" "Fortran compiler (ifort)"
check_command "g++" "C++ compiler (g++)" || check_command "clang++" "C++ compiler (clang++)"
echo ""

# Check optional tools
echo_check "Checking optional tools..."
check_command "python3" "Python 3" || check_command "python" "Python"
check_command "git" "Git"
echo ""

# Check Python packages (if Python available)
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    echo_check "Checking Python packages..."
    PYTHON_CMD=$(command -v python3 || command -v python)

    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
        echo_pass "PyTorch available: $TORCH_VERSION"
        ((PASS++))
    else
        echo_warn "PyTorch not installed (required for training models)"
        ((WARN++))
    fi

    if $PYTHON_CMD -c "import numpy" 2>/dev/null; then
        echo_pass "NumPy available"
        ((PASS++))
    else
        echo_warn "NumPy not installed"
        ((WARN++))
    fi
    echo ""
fi

# Code quality checks
echo_check "Running code quality checks..."

# Check for TODO/FIXME
TODO_COUNT=$(grep -r "TODO\|FIXME" src/ include/ 2>/dev/null | wc -l || echo 0)
if [ "$TODO_COUNT" -gt 0 ]; then
    echo_warn "Found $TODO_COUNT TODO/FIXME comments"
    ((WARN++))
else
    echo_pass "No TODO/FIXME comments found"
    ((PASS++))
fi

# Check for consistent file endings
echo_check "Checking file consistency..."
if file src/cpp/*.cpp | grep -q "CRLF"; then
    echo_warn "Some C++ files have CRLF line endings"
    ((WARN++))
else
    echo_pass "C++ files have Unix line endings"
    ((PASS++))
fi

if file src/fortran/*.f90 | grep -q "CRLF"; then
    echo_warn "Some Fortran files have CRLF line endings"
    ((WARN++))
else
    echo_pass "Fortran files have Unix line endings"
    ((PASS++))
fi
echo ""

# Summary
echo "======================================"
echo " Validation Summary"
echo "======================================"
echo -e "${GREEN}Passed:  $PASS${NC}"
echo -e "${YELLOW}Warnings: $WARN${NC}"
echo -e "${RED}Failed:  $FAIL${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo_fail "Validation failed! Please fix the errors above."
    exit 1
elif [ $WARN -gt 5 ]; then
    echo_warn "Validation passed with warnings. Consider addressing them."
    exit 0
else
    echo_pass "Validation successful!"
    echo ""
    echo "Framework structure is correct."
    echo "You can proceed with building:"
    echo "  ./scripts/build.sh"
    exit 0
fi
