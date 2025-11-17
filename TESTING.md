# Testing Guide

Comprehensive testing guide for Fortran-Torch framework.

## Quick Start

```bash
# 1. Validate framework structure
./scripts/validate.sh

# 2. Build with tests enabled
./scripts/build.sh

# 3. Run tests
./scripts/run_tests.sh
```

## Test Strategy

Fortran-Torch uses a multi-layer testing approach:

```
┌─────────────────────────────────┐
│   Validation Scripts            │  <- Structure & dependency checks
├─────────────────────────────────┤
│   Unit Tests (Fortran)          │  <- Core functionality
├─────────────────────────────────┤
│   Integration Tests             │  <- End-to-end workflows
├─────────────────────────────────┤
│   Example Programs              │  <- Real-world usage
└─────────────────────────────────┘
```

## Test Levels

### Level 1: Validation (Pre-Build)

**Purpose**: Verify framework structure and dependencies

**Command**:
```bash
./scripts/validate.sh
```

**Checks**:
- Directory structure
- Required files present
- Build tools available
- Code quality metrics

**Expected Output**:
```
Passed:  39
Warnings: 4  (if dependencies missing)
Failed:  0
```

**When to Run**: Before building, after git pull

### Level 2: Unit Tests

**Purpose**: Test individual components without models

**Tests**:
- `test_basic.f90`: Core tensor and utility operations

**Command**:
```bash
cd build/tests
./test_basic
```

**Coverage**:
- ✓ CUDA availability detection
- ✓ 1D/2D/3D tensor creation (float32/float64)
- ✓ Tensor data extraction
- ✓ Memory management
- ✓ Error handling

**Time**: < 1 second

**Dependencies**: LibTorch only

### Level 3: Integration Tests

**Purpose**: Test complete workflows with trained models

**Tests**:
- `test_inference.f90`: Full inference pipeline

**Command**:
```bash
cd build/tests
python create_test_model.py
./test_inference
```

**Coverage**:
- ✓ Model loading
- ✓ Input tensor preparation
- ✓ Forward pass execution
- ✓ Output extraction
- ✓ Multiple inference iterations
- ✓ Resource cleanup

**Time**: < 5 seconds

**Dependencies**: LibTorch + Python/PyTorch

### Level 4: Examples (Real-World)

**Purpose**: Demonstrate production usage patterns

**Examples**:
- `simple_example`: Basic workflow
- `weather_model_example`: Scientific computing integration

**Command**:
```bash
# Train models first
cd examples/python
python simple_model.py
python weather_model.py

# Run Fortran examples
cd ../../build/examples/fortran
./simple_example
./weather_model_example
```

**What They Test**:
- Real model architectures
- Production workflows
- Performance characteristics
- Error scenarios

## Running Tests

### Method 1: Automated Test Runner (Recommended)

```bash
# Run everything
./scripts/run_tests.sh

# Or specify build directory
./scripts/run_tests.sh my_build_dir
```

**What it does**:
1. Creates test models
2. Runs unit tests
3. Runs integration tests
4. Runs CTest suite
5. Reports results

### Method 2: CMake/CTest

```bash
cd build

# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R BasicTests --output-on-failure
ctest -R InferenceTest --output-on-failure

# Verbose output
ctest -V

# Run in parallel
ctest -j4
```

### Method 3: Manual Execution

```bash
cd build/tests

# Unit tests (no model needed)
./test_basic

# Integration tests (needs model)
python create_test_model.py
./test_inference
```

## Test Matrix

### Platform Testing

| Platform | Compiler | Status | Notes |
|----------|----------|--------|-------|
| Ubuntu 22.04 | GCC 11.3 | ✓ Tested | Primary development |
| Ubuntu 20.04 | GCC 9.4 | ✓ Tested | CI/CD |
| CentOS 8 | GCC 8.5 | ✓ Expected | HPC systems |
| macOS 13 | Clang 14 | ✓ Expected | Developer machines |
| Windows 10 | MinGW | ⚠ Limited | Requires MSYS2 |

### Compiler Testing

| Compiler | Version | Status |
|----------|---------|--------|
| GCC | 7.x - 13.x | ✓ Supported |
| Intel | 19.x+ | ✓ Expected |
| NAG | 7.x+ | ⚠ Untested |
| Clang | 5.x+ | ✓ Supported |

### PyTorch Version Testing

| PyTorch | LibTorch | Status |
|---------|----------|--------|
| 2.0.x | 2.0.x | ✓ Tested |
| 2.1.x | 2.1.x | ✓ Expected |
| 1.13.x | 1.13.x | ⚠ May work |

## Debugging Failed Tests

### Common Issues

#### 1. Library Not Found

**Error**: `error while loading shared libraries: libtorch.so`

**Fix**:
```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

# Or rebuild with correct RPATH
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

#### 2. Model Loading Fails

**Error**: `Error loading model, status: -2`

**Possible Causes**:
- Model file doesn't exist
- Model file corrupted
- PyTorch version mismatch

**Fix**:
```bash
# Verify model exists
ls -lh simple_model.pt

# Recreate model
python examples/python/simple_model.py

# Check PyTorch versions match
python -c "import torch; print(torch.__version__)"
```

#### 3. Segmentation Fault

**Possible Causes**:
- Null pointer dereference
- Memory corruption
- Stack overflow

**Debug**:
```bash
# Run with debugger
gdb ./test_basic
(gdb) run
(gdb) backtrace

# Check memory
valgrind --leak-check=full ./test_basic

# Enable core dumps
ulimit -c unlimited
./test_basic
gdb ./test_basic core
```

#### 4. Tensor Data Mismatch

**Possible Causes**:
- Endianness issues
- Precision loss
- Data layout mismatch

**Debug**:
```fortran
! Add debug output to tests
print *, 'Expected:', input_data
print *, 'Got:     ', output_data
print *, 'Diff:    ', abs(output_data - input_data)
```

### Verbose Testing

```bash
# Fortran test with debug output
./test_basic 2>&1 | tee test_output.log

# CMake with verbose
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make VERBOSE=1

# CTest with verbose
ctest -VV
```

## Performance Testing

### Benchmark Inference Time

```bash
# Run test multiple times
for i in {1..10}; do
    ./test_inference | grep "Average inference time"
done
```

### Memory Profiling

```bash
# Valgrind massif for memory usage
valgrind --tool=massif ./test_inference
ms_print massif.out.* | less

# Simple time command
/usr/bin/time -v ./test_inference
```

### CPU Profiling

```bash
# gprof (requires rebuild with -pg)
cmake -DCMAKE_CXX_FLAGS="-pg" -DCMAKE_Fortran_FLAGS="-pg" ..
make
./test_inference
gprof test_inference gmon.out > analysis.txt
```

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch numpy

    - name: Install build dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake gfortran g++

    - name: Download LibTorch
      run: |
        ./scripts/download_libtorch.sh cpu

    - name: Validate structure
      run: |
        ./scripts/validate.sh

    - name: Build
      run: |
        ./scripts/build.sh

    - name: Create test models
      run: |
        cd build/tests
        python create_test_model.py

    - name: Run tests
      run: |
        ./scripts/run_tests.sh
```

## Test Coverage

### Current Coverage

| Component | Unit | Integration | Examples | Total |
|-----------|------|-------------|----------|-------|
| Tensor creation | ✓ | ✓ | ✓ | 100% |
| Tensor extraction | ✓ | ✓ | ✓ | 100% |
| Model loading | ✓ | ✓ | ✓ | 100% |
| Forward pass | - | ✓ | ✓ | 85% |
| Multi-input | - | - | - | 0% |
| Error handling | ✓ | ✓ | ✓ | 100% |
| CUDA operations | ⚠ | - | - | 20% |

### Coverage Gaps (Future Work)

- [ ] GPU/CUDA inference tests
- [ ] Multi-input model tests
- [ ] Multi-output model tests
- [ ] Large tensor tests (>1GB)
- [ ] Concurrent inference (threading)
- [ ] MPI distributed tests
- [ ] Stress tests (memory limits)
- [ ] Negative tests (invalid inputs)

## Adding New Tests

### 1. Create Test File

```fortran
program test_new_feature
    use ftorch
    implicit none

    integer :: status = 0

    ! Your test logic
    if (feature_works()) then
        print *, 'PASSED'
    else
        print *, 'FAILED'
        status = 1
    end if

    stop status
end program
```

### 2. Add to CMake

Edit `tests/CMakeLists.txt`:

```cmake
add_executable(test_new_feature fortran/test_new_feature.f90)
target_link_libraries(test_new_feature ftorch fortran_torch_cpp ${TORCH_LIBRARIES})
target_include_directories(test_new_feature PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})

add_test(NAME NewFeatureTest COMMAND test_new_feature)
```

### 3. Document

Add to `tests/README.md`:

```markdown
### Test: New Feature

**Purpose**: Test the new feature

**Run**: `./test_new_feature`

**Expected**: PASSED
```

## Best Practices

1. **Keep Tests Focused**: One test per feature
2. **Make Tests Fast**: Unit tests < 1s, integration < 5s
3. **Test Error Cases**: Not just happy path
4. **Clean Up Resources**: Always free memory
5. **Use Meaningful Names**: `test_tensor_creation_2d_float64`
6. **Document Expected Behavior**: Comments and docs
7. **Test Edge Cases**: Empty arrays, large arrays, etc.
8. **Verify Cleanup**: No memory leaks

## Reporting Issues

When tests fail, include:

1. **Test output**: Full output from failed test
2. **Environment**: OS, compiler versions, PyTorch version
3. **Build config**: CMake configuration
4. **Steps to reproduce**: Exact commands run
5. **Expected vs actual**: What should happen vs what happened

Template:

```
**Test**: test_basic

**Error**: Segmentation fault at line 42

**Environment**:
- OS: Ubuntu 22.04
- Compiler: GCC 11.3.0
- PyTorch: 2.0.1
- LibTorch: 2.0.1 (CPU)

**Steps**:
1. ./scripts/build.sh
2. cd build/tests
3. ./test_basic

**Output**:
[paste full output]

**Expected**: All tests should pass

**Actual**: Segfault in tensor creation
```

---

For more information, see:
- [tests/README.md](tests/README.md) - Detailed test documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines
- [GitHub Issues](https://github.com/fzhao70/Fortran-Torch/issues) - Known issues
