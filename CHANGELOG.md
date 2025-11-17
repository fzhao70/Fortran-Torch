# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-17

### Added

#### Core Framework
- C++ wrapper library for PyTorch C++ API (libtorch)
- Fortran module with ISO_C_BINDING interfaces
- Type-safe Fortran API for loading and running PyTorch models
- Support for 1D, 2D, and 3D tensor operations
- Generic interfaces for float32 and float64 data types
- CPU and CUDA device support
- Comprehensive error handling and reporting

#### API Features
- `torch_load_model()` - Load TorchScript models
- `torch_free_model()` - Free model resources
- `torch_tensor_from_array()` - Create tensors from Fortran arrays
- `torch_tensor_to_array()` - Extract data from tensors to Fortran arrays
- `torch_forward()` - Run single-input inference
- `torch_forward_multi()` - Run multi-input inference
- `torch_cuda_available()` - Check CUDA availability
- `torch_free_tensor()` - Free tensor resources

#### Build System
- CMake-based build system
- Automatic LibTorch detection
- Example building infrastructure
- Installation support

#### Examples
- Simple neural network example (Python training + Fortran inference)
- Weather model parameterization example
- Complete workflow demonstration

#### Documentation
- Comprehensive README with quick start guide
- Detailed installation guide (INSTALL.md)
- Architecture documentation (ARCHITECTURE.md)
- Contributing guidelines (CONTRIBUTING.md)
- API reference in README
- In-code documentation comments

#### Tools & Scripts
- LibTorch download script
- Quick build script
- Python requirements file

#### Python Integration
- Example training scripts
- Model export utilities
- TorchScript conversion examples

### Features for Scientific Computing
- Weather and climate model integration examples
- Atmospheric state parameterization
- Performance timing utilities
- Batch processing patterns
- Grid-based computation examples

### Technical Details
- Zero Python runtime dependency
- Thread-local error message storage
- RAII-based memory management in C++
- Explicit resource cleanup in Fortran
- Data copying for memory safety
- Support for both CPU and GPU tensors

### Testing & Quality
- Example programs serve as integration tests
- Error handling throughout the stack
- Memory leak prevention
- Thread-safe error reporting

### Platform Support
- Linux (tested)
- macOS (supported)
- Windows with MinGW (supported)
- HPC systems with module environment (supported)

### Dependencies
- CMake >= 3.18
- C++17 compatible compiler
- Fortran compiler (gfortran, ifort, etc.)
- LibTorch (PyTorch C++)
- Python 3.7+ with PyTorch (for model training)

## [Unreleased]

### Planned
- Support for multiple model outputs
- Automatic batching utilities
- MPI-aware distributed inference
- Extended data type support (int32, int64)
- Comprehensive test suite
- Performance benchmarks
- Tutorial notebooks
- Model caching mechanisms

### Under Consideration
- Zero-copy tensor operations where possible
- Asynchronous inference
- Custom operator registration
- Quantized model support (int8 inference)
- Fortran-defined loss functions
- In-place tensor operations

---

## Version History

- **1.0.0** (2024-11-17): Initial release with core functionality

## Versioning

We use [Semantic Versioning](https://semver.org/):
- **Major version** (X.0.0): Incompatible API changes
- **Minor version** (0.X.0): Backwards-compatible new features
- **Patch version** (0.0.X): Backwards-compatible bug fixes

## Migration Guides

### Future Breaking Changes

When we release version 2.0.0, we will provide:
- Detailed migration guide
- Deprecation warnings in 1.x versions
- Support period for 1.x versions
- Example code updates

## Support

- Current stable: 1.0.0
- Development branch: main
- Long-term support: To be determined based on adoption

---

For detailed commit history, see the [GitHub repository](https://github.com/yourusername/Fortran-Torch/commits/main).
