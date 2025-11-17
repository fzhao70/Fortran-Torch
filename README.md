# Fortran-Torch

An easy-to-use framework for integrating PyTorch machine learning models into large Fortran projects, with a focus on scientific computing applications like weather and climate models.

## Features

- **Simple API**: Clean Fortran interface using ISO_C_BINDING
- **Zero Python Runtime**: Uses PyTorch C++ API (libtorch) - no Python interpreter needed at runtime
- **High Performance**: Minimal overhead for ML inference in Fortran applications
- **Type Safe**: Fortran module with type-safe tensor operations
- **GPU Support**: Optional CUDA support for accelerated inference
- **Production Ready**: Robust error handling and memory management

## Use Cases

Fortran-Torch is designed for integrating modern ML models into legacy Fortran codebases:

- **Weather & Climate Models**: ML-based parameterizations, bias correction, downscaling
- **Computational Fluid Dynamics**: Turbulence modeling, surrogate models
- **Finite Element Analysis**: Material property prediction, mesh refinement
- **Quantum Chemistry**: Potential energy surfaces, property prediction
- **Any Large Fortran Project**: Where you need ML inference without rewriting in Python

## Quick Start

### Prerequisites

- CMake >= 3.18
- C++ compiler with C++17 support (GCC >= 7, Clang >= 5)
- Fortran compiler (gfortran, ifort, etc.)
- PyTorch C++ library (libtorch)
- Python 3.7+ with PyTorch (for model training/export)

### Installation

1. **Download LibTorch**

```bash
# For CPU-only version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# For CUDA version (example for CUDA 12.1)
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

2. **Clone and Build**

```bash
git clone https://github.com/yourusername/Fortran-Torch.git
cd Fortran-Torch
mkdir build && cd build

# Configure with CMake (specify libtorch path)
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build
make -j$(nproc)

# Optional: Install
sudo make install
```

### Your First Example

1. **Train and export a PyTorch model**

```bash
cd examples/python
python simple_model.py
```

This creates `simple_model.pt` - a TorchScript model ready for Fortran.

2. **Run the Fortran example**

```bash
cd ../../build
./examples/fortran/simple_example
```

## Usage

### Training and Exporting Models (Python)

```python
import torch
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Train your model
model = MyModel()
# ... training code ...

# Export to TorchScript
model.eval()
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('my_model.pt')
```

### Using Models in Fortran

```fortran
program my_program
    use ftorch
    use iso_fortran_env, only: real32
    implicit none

    type(torch_model) :: model
    type(torch_tensor) :: input_tensor, output_tensor
    real(real32) :: input_data(10), output_data(5)

    ! Load model
    model = torch_load_model('my_model.pt', TORCH_DEVICE_CPU)

    ! Prepare input
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    ! Create tensor
    input_tensor = torch_tensor_from_array(input_data)

    ! Run inference
    output_tensor = torch_forward(model, input_tensor)

    ! Get results
    call torch_tensor_to_array(output_tensor, output_data)

    print *, 'Output:', output_data

    ! Cleanup
    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
    call torch_free_model(model)

end program my_program
```

## API Reference

### Model Operations

```fortran
! Load a TorchScript model
type(torch_model) function torch_load_model(model_path, device)
    character(len=*), intent(in) :: model_path
    integer(torch_device), intent(in), optional :: device

! Free a model
subroutine torch_free_model(model)
    type(torch_model), intent(inout) :: model

! Run inference
type(torch_tensor) function torch_forward(model, input)
    type(torch_model), intent(in) :: model
    type(torch_tensor), intent(in) :: input
```

### Tensor Operations

```fortran
! Create tensor from Fortran array (1D, 2D, or 3D)
type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real32/real64), intent(in) :: array(:) or array(:,:) or array(:,:,:)
    integer(torch_device), intent(in), optional :: device

! Copy tensor data to Fortran array
subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real32/real64), intent(out) :: array(:) or array(:,:) or array(:,:,:)

! Free a tensor
subroutine torch_free_tensor(tensor)
    type(torch_tensor), intent(inout) :: tensor
```

### Utility Functions

```fortran
! Check if CUDA is available
logical function torch_cuda_available()
```

### Constants

```fortran
! Data types
TORCH_FLOAT32  ! 32-bit floating point
TORCH_FLOAT64  ! 64-bit floating point
TORCH_INT32    ! 32-bit integer
TORCH_INT64    ! 64-bit integer

! Devices
TORCH_DEVICE_CPU   ! CPU device
TORCH_DEVICE_CUDA  ! CUDA GPU device
```

## Examples

### Simple Neural Network

See `examples/fortran/simple_example.f90` for a basic example of loading a model and running inference.

### Weather Model Integration

See `examples/fortran/weather_model_example.f90` for a realistic example of integrating ML parameterization into a weather model:

- Atmospheric state management
- Column-wise inference at grid points
- Applying ML-predicted tendencies
- Performance monitoring

## Building Your Project with Fortran-Torch

### Using CMake

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject Fortran)

# Find PyTorch
find_package(Torch REQUIRED)

# Find Fortran-Torch (if installed)
find_package(FortranTorch REQUIRED)

add_executable(my_app main.f90)
target_link_libraries(my_app
    ftorch
    fortran_torch_cpp
    ${TORCH_LIBRARIES}
)
```

### Using pkg-config (if installed)

```bash
gfortran my_app.f90 $(pkg-config --cflags --libs fortran-torch) -o my_app
```

## Performance Considerations

### CPU vs GPU

- For small models or infrequent calls: CPU is often faster (no transfer overhead)
- For large models or batch inference: GPU can provide significant speedup
- Profile your specific use case!

### Optimization Tips

1. **Batch Processing**: Process multiple grid points together when possible
2. **Model Optimization**: Use TorchScript optimization and quantization
3. **Memory Management**: Reuse tensors when possible, clean up promptly
4. **Threading**: libtorch can use OpenMP; set `OMP_NUM_THREADS` appropriately

### Benchmarking

The weather model example includes timing measurements:

```bash
./weather_model_example
# Outputs: average inference time per call
```

## Best Practices

### Model Design

1. **Keep it Simple**: Simpler models = faster inference
2. **Fixed Input Size**: TorchScript works best with fixed-size inputs
3. **Normalize Inputs**: Include normalization in the model when possible
4. **Test Thoroughly**: Validate against Python implementation

### Fortran Integration

1. **Error Handling**: Always check if pointers are associated
2. **Memory Management**: Free all tensors and models
3. **Thread Safety**: Create separate model instances per thread if using OpenMP
4. **Precision**: Match Fortran precision (real32/real64) with PyTorch dtype

### Deployment

1. **Bundle libtorch**: Include libtorch libraries with your distribution
2. **RPATH**: Set correctly so executables find libtorch
3. **Testing**: Test on target architecture before deployment
4. **Documentation**: Document model versions and expected inputs/outputs

## Troubleshooting

### Build Issues

**Problem**: Cannot find libtorch
```
Solution: Set CMAKE_PREFIX_PATH to libtorch directory
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

**Problem**: Undefined references to torch symbols
```
Solution: Check that libtorch ABI matches your compiler
Download the cxx11-abi version for GCC >= 5
```

### Runtime Issues

**Problem**: Model fails to load
```
Solution:
1. Ensure model file exists and path is correct
2. Check model was exported with compatible PyTorch version
3. Verify model was traced/scripted correctly
```

**Problem**: Segmentation fault
```
Solution:
1. Check all tensors are properly allocated
2. Verify array dimensions match expected input shape
3. Ensure proper cleanup (no double-free)
```

### CUDA Issues

**Problem**: CUDA out of memory
```
Solution:
1. Reduce batch size
2. Use smaller model
3. Clear unused tensors promptly
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use Fortran-Torch in your research, please cite:

```bibtex
@software{fortran_torch,
  title = {Fortran-Torch: PyTorch Integration for Fortran},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Fortran-Torch}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent C++ API
- Fortran community for ISO_C_BINDING standardization
- Scientific computing community for testing and feedback

## Related Projects

- [FTorch](https://github.com/Cambridge-ICCS/FTorch) - Similar project from Cambridge ICCS
- [pytorch-fortran](https://github.com/alexeedm/pytorch-fortran) - Alternative PyTorch-Fortran binding
- [TensorFlow-Fortran](https://github.com/scientific-computing/tensorflow-fortran) - TensorFlow bindings for Fortran

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Email**: your.email@example.com

## Roadmap

- [ ] Support for multiple outputs
- [ ] Support for multiple model instances
- [ ] Automatic batching utilities
- [ ] MPI-aware distributed inference
- [ ] Model caching and optimization
- [ ] Extended dtype support
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] Tutorial notebooks

---

**Happy Computing! 🚀**
