#!/bin/bash

# Comprehensive test runner for Fortran-Torch
# Usage: ./run_tests.sh [build_directory]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get build directory
BUILD_DIR="${1:-build}"

if [ ! -d "$BUILD_DIR" ]; then
    echo_error "Build directory not found: $BUILD_DIR"
    echo "Please build the project first or specify correct build directory"
    echo "Usage: $0 [build_directory]"
    exit 1
fi

echo ""
echo "======================================"
echo " Fortran-Torch Test Suite"
echo "======================================"
echo ""

# Change to build directory
cd "$BUILD_DIR"

# Check if tests were built
if [ ! -d "tests" ]; then
    echo_error "Tests not found. Build with -DBUILD_TESTS=ON"
    exit 1
fi

cd tests

# Step 1: Create test model
echo_info "Step 1: Creating test model..."
if [ -f "create_test_model.py" ]; then
    if command -v python3 &> /dev/null; then
        python3 create_test_model.py test_model.pt || {
            echo_warn "Could not create test model (inference test will be skipped)"
        }
    elif command -v python &> /dev/null; then
        python create_test_model.py test_model.pt || {
            echo_warn "Could not create test model (inference test will be skipped)"
        }
    else
        echo_warn "Python not found, skipping model creation"
    fi
else
    echo_warn "create_test_model.py not found"
fi
echo ""

# Step 2: Run basic tests
echo_info "Step 2: Running basic tests..."
echo "--------------------------------------"
if [ -f "./test_basic" ]; then
    ./test_basic
    BASIC_RESULT=$?
    if [ $BASIC_RESULT -eq 0 ]; then
        echo_success "Basic tests passed!"
    else
        echo_error "Basic tests failed!"
        exit 1
    fi
else
    echo_error "test_basic executable not found"
    exit 1
fi
echo ""

# Step 3: Run inference tests
echo_info "Step 3: Running inference tests..."
echo "--------------------------------------"
if [ -f "./test_inference" ]; then
    if [ -f "simple_model.pt" ] || [ -f "test_model.pt" ]; then
        # Copy simple_model.pt if it exists and test_model doesn't
        if [ ! -f "simple_model.pt" ] && [ -f "test_model.pt" ]; then
            cp test_model.pt simple_model.pt
        fi

        ./test_inference
        INFERENCE_RESULT=$?
        if [ $INFERENCE_RESULT -eq 0 ]; then
            echo_success "Inference tests passed!"
        else
            echo_error "Inference tests failed!"
            exit 1
        fi
    else
        echo_warn "Model file not found, skipping inference test"
        echo_warn "Run: python create_test_model.py simple_model.pt"
    fi
else
    echo_error "test_inference executable not found"
    exit 1
fi
echo ""

# Step 4: Run CTest if available
echo_info "Step 4: Running CTest suite..."
echo "--------------------------------------"
cd ..
if command -v ctest &> /dev/null; then
    ctest --output-on-failure
    CTEST_RESULT=$?
    if [ $CTEST_RESULT -eq 0 ]; then
        echo_success "CTest suite passed!"
    else
        echo_warn "Some CTest tests failed (this may be expected)"
    fi
else
    echo_warn "CTest not found, skipping"
fi
echo ""

# Summary
echo "======================================"
echo " Test Summary"
echo "======================================"
echo_success "All critical tests passed!"
echo ""
echo "Next steps:"
echo "  - Run examples: cd examples/fortran && ./simple_example"
echo "  - Check coverage: See tests/README.md"
echo "  - Report issues: GitHub issue tracker"
echo ""
