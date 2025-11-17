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
git clone https://github.com/fzhao70/Fortran-Torch.git
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

## Weather Model Integration

Fortran-Torch is specifically designed for integrating ML into operational weather and climate models. We provide comprehensive integration guides for:

### Supported Models

- **[WRF (Weather Research and Forecasting)](docs/INTEGRATION_WRF.md)** - Regional weather model with modular physics
- **[MPAS (Model for Prediction Across Scales)](docs/INTEGRATION_MPAS.md)** - Variable-resolution global model
- **[FV3GFS (Finite-Volume Cubed-Sphere)](docs/INTEGRATION_FV3.md)** - NOAA's operational global forecast system

### Quick Links

- **[General Weather Model Integration Guide](docs/WEATHER_MODEL_INTEGRATION.md)** - Patterns, best practices, and common integration approaches
- **[WRF Integration](docs/INTEGRATION_WRF.md)** - Detailed WRF integration with physics schemes, build configuration, and examples
- **[MPAS Integration](docs/INTEGRATION_MPAS.md)** - Unstructured mesh handling, variable resolution, and MPAS-specific patterns
- **[FV3 Integration](docs/INTEGRATION_FV3.md)** - CCPP-compliant schemes, cubed-sphere considerations, operational deployment

### Integration Examples

**ML-based Convection Scheme for WRF:**
```fortran
use module_cu_ml
use ftorch

! Initialize ML model
call cu_ml_init('convection_model.pt')

! In physics loop
call cu_ml_driver(t3d, qv3d, p3d, rthcuten, rqvcuten, ...)
```

**Scale-Aware Parameterization for MPAS:**
```fortran
! Include mesh resolution for variable-resolution grids
do iCell = 1, nCells
    resolution = get_cell_resolution(iCell)
    call ml_convection(theta(:,iCell), resolution, tend(:,iCell))
end do
```

**CCPP-Compliant Scheme for FV3:**
```fortran
! CCPP metadata-driven integration
subroutine ml_convection_run(t, q, prsl, dt_t, dq_v, errmsg, errflg)
    ! Standard CCPP interface
    ! Operational-ready deployment
end subroutine
```

### Use Cases

1. **Physics Parameterizations** - Replace or augment convection, PBL, microphysics
2. **Bias Correction** - Correct systematic model errors
3. **Subgrid Processes** - Scale-aware turbulence, clouds
4. **Data Assimilation** - ML observation operators
5. **Post-Processing** - Downscaling, ensemble calibration

See the [Weather Model Integration Guide](docs/WEATHER_MODEL_INTEGRATION.md) for comprehensive documentation.

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

The Fortran-Torch API provides a type-safe, easy-to-use interface for PyTorch inference in Fortran. All functions use ISO_C_BINDING for seamless interoperability.

### Core Types

#### torch_model

Opaque type representing a loaded PyTorch model.

```fortran
type :: torch_model
    type(c_ptr) :: ptr = c_null_ptr
end type torch_model
```

**Usage:**
- Created by `torch_load_model()`
- Used for inference with `torch_forward()`
- Must be freed with `torch_free_model()`
- Check validity: `c_associated(model%ptr)`

#### torch_tensor

Opaque type representing a PyTorch tensor.

```fortran
type :: torch_tensor
    type(c_ptr) :: ptr = c_null_ptr
end type torch_tensor
```

**Usage:**
- Created by `torch_tensor_from_array()` or `torch_forward()`
- Converted to Fortran arrays with `torch_tensor_to_array()`
- Must be freed with `torch_free_tensor()`
- Check validity: `c_associated(tensor%ptr)`

### Model Operations

#### torch_load_model

Load a TorchScript model from a file.

```fortran
type(torch_model) function torch_load_model(model_path, device)
    character(len=*), intent(in) :: model_path
    integer(torch_device), intent(in), optional :: device
end function torch_load_model
```

**Parameters:**
- `model_path`: Path to the `.pt` TorchScript model file
- `device` (optional): Device to load model on (`TORCH_DEVICE_CPU` or `TORCH_DEVICE_CUDA`). Defaults to CPU.

**Returns:**
- `torch_model`: Model handle. Check `c_associated(model%ptr)` for success.

**Example:**
```fortran
type(torch_model) :: model

! Load on CPU
model = torch_load_model('my_model.pt')

! Load on GPU
model = torch_load_model('my_model.pt', TORCH_DEVICE_CUDA)

! Check if loaded successfully
if (.not. c_associated(model%ptr)) then
    print *, "Error loading model"
    stop 1
end if
```

**Notes:**
- Model file must exist and be a valid TorchScript model
- For CUDA models, ensure CUDA is available (`torch_cuda_available()`)
- Model remains in memory until `torch_free_model()` is called

#### torch_forward

Run inference on a loaded model.

```fortran
type(torch_tensor) function torch_forward(model, input)
    type(torch_model), intent(in) :: model
    type(torch_tensor), intent(in) :: input
end function torch_forward
```

**Parameters:**
- `model`: Loaded model from `torch_load_model()`
- `input`: Input tensor from `torch_tensor_from_array()`

**Returns:**
- `torch_tensor`: Output tensor. Check `c_associated(output%ptr)` for success.

**Example:**
```fortran
type(torch_model) :: model
type(torch_tensor) :: input_tensor, output_tensor
real(real32) :: input_data(10), output_data(5)

model = torch_load_model('my_model.pt')
input_tensor = torch_tensor_from_array(input_data)

! Run inference
output_tensor = torch_forward(model, input_tensor)

! Extract results
call torch_tensor_to_array(output_tensor, output_data)

! Cleanup
call torch_free_tensor(output_tensor)
```

**Notes:**
- Input tensor shape must match model's expected input
- Output tensor must be freed after use
- Thread-safe if using separate model instances per thread

#### torch_free_model

Free a loaded model and release resources.

```fortran
subroutine torch_free_model(model)
    type(torch_model), intent(inout) :: model
end subroutine torch_free_model
```

**Parameters:**
- `model`: Model to free (pointer set to `c_null_ptr` after freeing)

**Example:**
```fortran
type(torch_model) :: model

model = torch_load_model('my_model.pt')
! ... use model ...
call torch_free_model(model)
```

**Notes:**
- Safe to call multiple times (checks for null pointer)
- Should be called before program termination
- Model cannot be used after freeing

### Tensor Operations

#### torch_tensor_from_array

Create a PyTorch tensor from a Fortran array. Generic interface supporting multiple ranks and types.

```fortran
! Generic interface
interface torch_tensor_from_array
    module procedure torch_tensor_from_array_real32_1d
    module procedure torch_tensor_from_array_real32_2d
    module procedure torch_tensor_from_array_real32_3d
    module procedure torch_tensor_from_array_real64_1d
    module procedure torch_tensor_from_array_real64_2d
    module procedure torch_tensor_from_array_real64_3d
end interface torch_tensor_from_array
```

**Supported Signatures:**

**1D Arrays:**
```fortran
type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real32), intent(in) :: array(:)
    integer(torch_device), intent(in), optional :: device
end function

type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real64), intent(in) :: array(:)
    integer(torch_device), intent(in), optional :: device
end function
```

**2D Arrays:**
```fortran
type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real32), intent(in) :: array(:,:)
    integer(torch_device), intent(in), optional :: device
end function

type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real64), intent(in) :: array(:,:)
    integer(torch_device), intent(in), optional :: device
end function
```

**3D Arrays:**
```fortran
type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real32), intent(in) :: array(:,:,:)
    integer(torch_device), intent(in), optional :: device
end function

type(torch_tensor) function torch_tensor_from_array(array, device)
    real(real64), intent(in) :: array(:,:,:)
    integer(torch_device), intent(in), optional :: device
end function
```

**Parameters:**
- `array`: Fortran array (1D, 2D, or 3D; float32 or float64)
- `device` (optional): Device to create tensor on. Defaults to CPU.

**Returns:**
- `torch_tensor`: Tensor containing copy of array data

**Examples:**

```fortran
! 1D array (float32)
real(real32) :: vec(100)
type(torch_tensor) :: tensor1d
vec = 1.0
tensor1d = torch_tensor_from_array(vec)

! 2D array (float64)
real(real64) :: mat(50, 20)
type(torch_tensor) :: tensor2d
mat = 2.0d0
tensor2d = torch_tensor_from_array(mat)

! 3D array on GPU
real(real32) :: cube(10, 10, 5)
type(torch_tensor) :: tensor3d_gpu
cube = 3.0
tensor3d_gpu = torch_tensor_from_array(cube, TORCH_DEVICE_CUDA)

! Cleanup
call torch_free_tensor(tensor1d)
call torch_free_tensor(tensor2d)
call torch_free_tensor(tensor3d_gpu)
```

**Notes:**
- Data is copied from Fortran to PyTorch memory
- Fortran column-major order is preserved in tensor
- Allocatable/pointer arrays are supported (must be allocated)
- Created tensors must be freed with `torch_free_tensor()`

#### torch_tensor_to_array

Copy data from a PyTorch tensor to a Fortran array. Generic interface for multiple ranks and types.

```fortran
! Generic interface
interface torch_tensor_to_array
    module procedure torch_tensor_to_array_real32_1d
    module procedure torch_tensor_to_array_real32_2d
    module procedure torch_tensor_to_array_real32_3d
    module procedure torch_tensor_to_array_real64_1d
    module procedure torch_tensor_to_array_real64_2d
    module procedure torch_tensor_to_array_real64_3d
end interface torch_tensor_to_array
```

**Supported Signatures:**

**1D Arrays:**
```fortran
subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real32), intent(out) :: array(:)
end subroutine

subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real64), intent(out) :: array(:)
end subroutine
```

**2D Arrays:**
```fortran
subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real32), intent(out) :: array(:,:)
end subroutine

subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real64), intent(out) :: array(:,:)
end subroutine
```

**3D Arrays:**
```fortran
subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real32), intent(out) :: array(:,:,:)
end subroutine

subroutine torch_tensor_to_array(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(real64), intent(out) :: array(:,:,:)
end subroutine
```

**Parameters:**
- `tensor`: Source tensor from `torch_forward()` or `torch_tensor_from_array()`
- `array`: Destination Fortran array (must be pre-allocated with correct shape)

**Examples:**

```fortran
type(torch_tensor) :: output_tensor
real(real32) :: output_1d(10)
real(real64) :: output_2d(5, 20)
real(real32) :: output_3d(8, 8, 16)

! Assuming output_tensor is from model inference

! Extract to 1D array
call torch_tensor_to_array(output_tensor, output_1d)

! Extract to 2D array
call torch_tensor_to_array(output_tensor, output_2d)

! Extract to 3D array
call torch_tensor_to_array(output_tensor, output_3d)
```

**Notes:**
- Array must be pre-allocated with correct dimensions
- Dimensions must exactly match tensor shape
- Data is copied from PyTorch to Fortran memory
- Type conversion (float32 ↔ float64) is handled automatically
- Tensor remains valid after extraction (still needs to be freed)

#### torch_free_tensor

Free a tensor and release resources.

```fortran
subroutine torch_free_tensor(tensor)
    type(torch_tensor), intent(inout) :: tensor
end subroutine torch_free_tensor
```

**Parameters:**
- `tensor`: Tensor to free (pointer set to `c_null_ptr` after freeing)

**Example:**
```fortran
type(torch_tensor) :: input_tensor, output_tensor

input_tensor = torch_tensor_from_array(input_data)
output_tensor = torch_forward(model, input_tensor)

! Use tensors...

! Cleanup
call torch_free_tensor(input_tensor)
call torch_free_tensor(output_tensor)
```

**Notes:**
- Safe to call multiple times (checks for null pointer)
- Should be called for all created tensors
- Tensor cannot be used after freeing

### Utility Functions

#### torch_cuda_available

Check if CUDA (GPU) support is available.

```fortran
logical function torch_cuda_available()
end function torch_cuda_available
```

**Returns:**
- `.true.` if CUDA is available
- `.false.` if CUDA is not available (CPU-only build or no GPU)

**Example:**
```fortran
logical :: has_cuda
integer(torch_device) :: device

has_cuda = torch_cuda_available()

if (has_cuda) then
    print *, "CUDA is available, using GPU"
    device = TORCH_DEVICE_CUDA
else
    print *, "CUDA not available, using CPU"
    device = TORCH_DEVICE_CPU
end if

model = torch_load_model('model.pt', device)
```

**Notes:**
- Returns false if LibTorch is built without CUDA
- Returns false if no CUDA-capable GPU is detected
- Check before attempting to use `TORCH_DEVICE_CUDA`

#### torch_get_last_error

Get the last error message from the C++ layer.

```fortran
function torch_get_last_error() result(error_msg)
    character(len=512) :: error_msg
end function torch_get_last_error
```

**Returns:**
- Character string with last error message (empty if no error)

**Example:**
```fortran
type(torch_model) :: model

model = torch_load_model('nonexistent.pt')

if (.not. c_associated(model%ptr)) then
    print *, "Error: ", trim(torch_get_last_error())
    stop 1
end if
```

**Notes:**
- Error messages are thread-local
- Error is cleared after retrieval
- Useful for debugging model loading and inference issues

### Constants and Enumerations

#### Data Types

```fortran
! Tensor data types (torch_dtype)
integer(c_int), parameter :: TORCH_FLOAT32 = 0  ! 32-bit floating point
integer(c_int), parameter :: TORCH_FLOAT64 = 1  ! 64-bit floating point
integer(c_int), parameter :: TORCH_INT32   = 2  ! 32-bit integer
integer(c_int), parameter :: TORCH_INT64   = 3  ! 64-bit integer
```

**Usage:**
- Automatically determined from Fortran array type
- Generally not used directly (handled by generic interfaces)

**Type Mapping:**
- `real(real32)` → `TORCH_FLOAT32` → PyTorch `float32`
- `real(real64)` → `TORCH_FLOAT64` → PyTorch `float64`

#### Devices

```fortran
! Device types (torch_device)
integer(c_int), parameter :: TORCH_DEVICE_CPU  = 0  ! CPU device
integer(c_int), parameter :: TORCH_DEVICE_CUDA = 1  ! CUDA GPU device
```

**Usage:**
```fortran
! Load model on CPU
model = torch_load_model('model.pt', TORCH_DEVICE_CPU)

! Load model on GPU (if available)
if (torch_cuda_available()) then
    model = torch_load_model('model.pt', TORCH_DEVICE_CUDA)
end if

! Create tensor on GPU
tensor = torch_tensor_from_array(data, TORCH_DEVICE_CUDA)
```

**Notes:**
- Default device is CPU if not specified
- Model and tensors should be on the same device
- CUDA device requires CUDA-enabled LibTorch build

### Complete Usage Example

Here's a comprehensive example demonstrating the full API:

```fortran
program complete_api_example
    use ftorch
    use iso_fortran_env, only: real32, real64
    implicit none

    ! Variables
    type(torch_model) :: model
    type(torch_tensor) :: input_tensor, output_tensor
    real(real32), allocatable :: input_data(:,:)
    real(real32), allocatable :: output_data(:)
    integer :: i, j
    logical :: cuda_available
    integer(torch_device) :: device
    character(len=512) :: error_msg

    ! Check CUDA availability
    cuda_available = torch_cuda_available()
    if (cuda_available) then
        print *, "CUDA available - using GPU"
        device = TORCH_DEVICE_CUDA
    else
        print *, "CUDA not available - using CPU"
        device = TORCH_DEVICE_CPU
    end if

    ! Allocate input data (batch of 32, features of 128)
    allocate(input_data(32, 128))
    allocate(output_data(10))

    ! Initialize input
    do j = 1, 128
        do i = 1, 32
            input_data(i, j) = real(i + j, real32)
        end do
    end do

    ! Load model
    model = torch_load_model('classifier.pt', device)
    if (.not. c_associated(model%ptr)) then
        error_msg = torch_get_last_error()
        print *, "Failed to load model: ", trim(error_msg)
        stop 1
    end if
    print *, "Model loaded successfully"

    ! Create input tensor
    input_tensor = torch_tensor_from_array(input_data, device)
    if (.not. c_associated(input_tensor%ptr)) then
        print *, "Failed to create input tensor"
        call torch_free_model(model)
        stop 1
    end if

    ! Run inference
    output_tensor = torch_forward(model, input_tensor)
    if (.not. c_associated(output_tensor%ptr)) then
        error_msg = torch_get_last_error()
        print *, "Inference failed: ", trim(error_msg)
        call torch_free_tensor(input_tensor)
        call torch_free_model(model)
        stop 1
    end if
    print *, "Inference completed"

    ! Extract results
    call torch_tensor_to_array(output_tensor, output_data)

    print *, "Output:", output_data

    ! Cleanup - IMPORTANT!
    call torch_free_tensor(input_tensor)
    call torch_free_tensor(output_tensor)
    call torch_free_model(model)

    deallocate(input_data)
    deallocate(output_data)

    print *, "Cleanup completed"

end program complete_api_example
```

### Error Handling Best Practices

Always check return values and handle errors:

```fortran
! 1. Check model loading
model = torch_load_model('model.pt')
if (.not. c_associated(model%ptr)) then
    print *, "ERROR: ", trim(torch_get_last_error())
    stop 1
end if

! 2. Check tensor creation
tensor = torch_tensor_from_array(data)
if (.not. c_associated(tensor%ptr)) then
    print *, "ERROR: Failed to create tensor"
    call torch_free_model(model)
    stop 1
end if

! 3. Check inference
output = torch_forward(model, input)
if (.not. c_associated(output%ptr)) then
    print *, "ERROR: Inference failed"
    call torch_free_tensor(input)
    call torch_free_model(model)
    stop 1
end if

! 4. Always cleanup
call torch_free_tensor(input)
call torch_free_tensor(output)
call torch_free_model(model)
```

### Thread Safety

For OpenMP parallel regions, use thread-private model instances:

```fortran
!$OMP PARALLEL PRIVATE(model, input_tensor, output_tensor)
    ! Each thread loads its own model
    model = torch_load_model('model.pt')

    !$OMP DO
    do i = 1, n_samples
        input_tensor = torch_tensor_from_array(data(:,i))
        output_tensor = torch_forward(model, input_tensor)
        call torch_tensor_to_array(output_tensor, results(:,i))
        call torch_free_tensor(input_tensor)
        call torch_free_tensor(output_tensor)
    end do
    !$OMP END DO

    call torch_free_model(model)
!$OMP END PARALLEL
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
  author = {Fanghe Zhao},
  year = {2024},
  url = {https://github.com/fzhao70/Fortran-Torch}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent C++ API
- Fortran community for ISO_C_BINDING standardization
- Scientific computing community for testing and feedback

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions

---

**Happy Computing! 🚀**
