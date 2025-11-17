# Fortran-Torch Tests

This directory contains the test suite for Fortran-Torch.

## Test Structure

```
tests/
├── CMakeLists.txt              # Test build configuration
├── README.md                   # This file
├── fortran/                    # Fortran tests
│   ├── test_basic.f90         # Basic unit tests
│   └── test_inference.f90     # End-to-end inference test
├── python/                     # Python test utilities
│   └── create_test_model.py  # Create test models
└── scripts/                    # Test automation scripts
    └── run_all_tests.sh       # Run complete test suite
```

## Test Categories

### 1. Basic Tests (`test_basic.f90`)

Unit tests for core functionality without requiring a trained model:

- **Test 1**: CUDA availability check
- **Test 2**: 1D tensor creation/extraction (float32)
- **Test 3**: 2D tensor creation/extraction (float32)
- **Test 4**: 3D tensor creation/extraction (float32)
- **Test 5**: 1D tensor creation/extraction (float64)
- **Test 6**: 2D tensor creation/extraction (float64)
- **Test 7**: Model loading (if test_model.pt exists)

**Run:**
```bash
cd build/tests
./test_basic
```

**Expected output:**
```
======================================
Fortran-Torch Basic Tests
======================================

Test 1: CUDA availability check
  CUDA available:  F
  PASSED

Test 2: 1D tensor (float32) creation and extraction
  PASSED

Test 3: 2D tensor (float32) creation and extraction
  PASSED

...

======================================
Test Summary
======================================
Total tests:   7
Passed:        7
Failed:        0

RESULT: ALL TESTS PASSED
```

### 2. Inference Tests (`test_inference.f90`)

End-to-end integration test requiring a trained model:

- Model loading
- Tensor creation from Fortran arrays
- Forward pass execution
- Result extraction
- Multiple inference iterations
- Resource cleanup

**Prerequisites:**
```bash
# Create test model
cd examples/python
python simple_model.py
cp simple_model.pt ../../build/tests/
```

**Run:**
```bash
cd build/tests
./test_inference
```

## Running Tests

### Method 1: Using CTest (Recommended)

```bash
# Build the project with tests
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DBUILD_TESTS=ON ..
make -j$(nproc)

# Create test model (required for inference test)
cd tests
python create_test_model.py

# Run all tests
ctest --output-on-failure

# Or run specific test
ctest -R BasicTests --output-on-failure
ctest -R InferenceTest --output-on-failure
```

### Method 2: Direct Execution

```bash
cd build/tests

# Run basic tests (no model required)
./test_basic

# Run inference test (requires model)
python create_test_model.py
./test_inference
```

### Method 3: Using Make Targets

```bash
cd build

# Create test model
make create_test_model

# Run all tests
make test
# or
make run_tests
```

## Test Requirements

### Basic Tests
- **Dependencies**: LibTorch, Fortran-Torch libraries
- **Models**: None required
- **Runtime**: < 1 second

### Inference Tests
- **Dependencies**: LibTorch, Fortran-Torch libraries, Python with PyTorch
- **Models**: simple_model.pt (created by Python script)
- **Runtime**: < 5 seconds

## Creating Test Models

### Simple Test Model

The `create_test_model.py` script creates a minimal model:

```bash
python tests/python/create_test_model.py output_path.pt
```

**Model specifications:**
- Input: (batch_size, 10)
- Output: (batch_size, 5)
- Architecture: Single linear layer
- Weights: Initialized to 0.1 (for reproducibility)

### Custom Test Models

To create custom test models:

```python
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your model

    def forward(self, x):
        # Define forward pass
        return output

# Create and export
model = YourModel()
model.eval()
example = torch.randn(1, input_size)
traced = torch.jit.trace(model, example)
traced.save('your_model.pt')
```

## Continuous Integration

### GitHub Actions

The `.github/workflows/test.yml` configuration runs tests automatically:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Download LibTorch
        run: ./scripts/download_libtorch.sh cpu
      - name: Build
        run: ./scripts/build.sh
      - name: Create test model
        run: cd build/tests && python create_test_model.py
      - name: Run tests
        run: cd build && ctest --output-on-failure
```

## Debugging Failed Tests

### Check LibTorch Path

```bash
# Verify libtorch is found
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .. 2>&1 | grep Torch

# Expected output:
# -- Found Torch: /path/to/libtorch/lib/libtorch.so
```

### Check Library Paths

```bash
# Linux/macOS
ldd ./test_basic | grep torch
# or
otool -L ./test_basic | grep torch

# Should show libtorch libraries
```

### Run with Debugging

```bash
# Run with verbose output
./test_basic

# Run with debugger
gdb ./test_basic
(gdb) run

# Check for memory leaks
valgrind --leak-check=full ./test_basic
```

### Common Issues

**Issue**: `error while loading shared libraries: libtorch.so`
```bash
# Solution: Add libtorch to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

**Issue**: `test_inference` can't find model
```bash
# Solution: Create model in test directory
cd build/tests
python create_test_model.py
```

**Issue**: Tests fail with segmentation fault
```bash
# Possible causes:
# 1. LibTorch version mismatch - rebuild with same version
# 2. Memory corruption - check with valgrind
# 3. Null pointer - check model/tensor creation succeeded
```

## Adding New Tests

### 1. Create Test File

Create `test_new_feature.f90`:

```fortran
program test_new_feature
    use ftorch
    implicit none

    ! Your test code here

    if (test_passed) then
        print *, 'PASSED'
    else
        print *, 'FAILED'
        stop 1
    end if
end program
```

### 2. Update CMakeLists.txt

Add to `tests/CMakeLists.txt`:

```cmake
add_executable(test_new_feature fortran/test_new_feature.f90)
target_link_libraries(test_new_feature ftorch fortran_torch_cpp ${TORCH_LIBRARIES})
target_include_directories(test_new_feature PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})

add_test(NAME NewFeatureTest COMMAND test_new_feature)
```

### 3. Rebuild and Test

```bash
cd build
make
ctest -R NewFeatureTest --output-on-failure
```

## Performance Testing

### Benchmarking Inference

```fortran
! Time multiple inferences
call cpu_time(start)
do i = 1, n_iterations
    output = torch_forward(model, input)
end do
call cpu_time(end)

avg_time = (end - start) / n_iterations
print *, 'Average inference time:', avg_time * 1000, 'ms'
```

### Memory Usage

```bash
# Monitor memory during test
/usr/bin/time -v ./test_inference

# Expected output includes:
# Maximum resident set size (kbytes): XXXX
```

## Test Coverage

Current test coverage:

| Component | Coverage | Tests |
|-----------|----------|-------|
| Tensor creation | ✓ | test_basic |
| Tensor extraction | ✓ | test_basic |
| Model loading | ✓ | test_basic, test_inference |
| Forward pass | ✓ | test_inference |
| Error handling | ✓ | test_basic |
| Resource cleanup | ✓ | test_inference |
| Multi-dimensional arrays | ✓ | test_basic |
| Different data types | ✓ | test_basic |

## Future Tests

Planned additions:

- [ ] GPU/CUDA tests
- [ ] Multi-input model tests
- [ ] Multi-output model tests
- [ ] Large tensor tests (>1GB)
- [ ] Stress tests (many iterations)
- [ ] Thread safety tests (OpenMP)
- [ ] MPI distributed tests
- [ ] Performance benchmarks

## Reporting Issues

If tests fail:

1. Check test output for specific error
2. Verify prerequisites (LibTorch, model files)
3. Run with debugging enabled
4. Check GitHub issues for similar problems
5. Create new issue with:
   - Test output
   - Environment details (OS, compiler versions)
   - Build configuration
   - Steps to reproduce

## Contributing Tests

We welcome test contributions! Please:

1. Follow existing test structure
2. Include documentation
3. Ensure tests pass on clean system
4. Add to this README
5. Submit pull request

---

For questions about testing, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
