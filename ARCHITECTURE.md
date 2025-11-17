# Fortran-Torch Architecture

This document describes the architecture and design decisions of the Fortran-Torch framework.

## Overview

Fortran-Torch provides a bridge between Fortran code and PyTorch models by creating a layered architecture:

```
┌─────────────────────────────────────────┐
│       Fortran Application Code          │
│    (Weather Models, CFD, etc.)          │
└──────────────┬──────────────────────────┘
               │
               │ Use ftorch module
               ▼
┌─────────────────────────────────────────┐
│      Fortran Module (ftorch.f90)        │
│   - Type-safe wrappers                  │
│   - Generic interfaces                  │
│   - ISO_C_BINDING                       │
└──────────────┬──────────────────────────┘
               │
               │ C interface
               ▼
┌─────────────────────────────────────────┐
│    C++ Library (fortran_torch.cpp)      │
│   - C API implementation                │
│   - Error handling                      │
│   - Memory management                   │
└──────────────┬──────────────────────────┘
               │
               │ LibTorch API
               ▼
┌─────────────────────────────────────────┐
│         LibTorch (PyTorch C++)          │
│   - Model execution                     │
│   - Tensor operations                   │
│   - GPU support                         │
└─────────────────────────────────────────┘
```

## Components

### 1. Fortran Module (`ftorch.f90`)

**Purpose**: Provide a natural, type-safe Fortran interface

**Key Features**:
- Opaque derived types for models and tensors
- Generic interfaces for different array ranks and types
- Automatic C string handling
- Fortran-style error reporting

**Design Decisions**:
- Use `iso_c_binding` for C interoperability
- Hide C pointers behind Fortran types
- Provide overloaded interfaces for convenience
- Keep API simple and intuitive for Fortran programmers

**Types**:
```fortran
type :: torch_model
    type(c_ptr) :: ptr = c_null_ptr
end type

type :: torch_tensor
    type(c_ptr) :: ptr = c_null_ptr
end type
```

### 2. C++ Library (`fortran_torch.cpp`)

**Purpose**: Bridge between C calling convention and C++ PyTorch API

**Key Features**:
- C-compatible API (extern "C")
- Exception handling (C++ to C error codes)
- Resource management (RAII internally)
- Thread-local error messages

**Design Decisions**:
- All functions return status codes
- Use opaque pointers for object handles
- Convert C++ exceptions to error codes
- Store detailed error messages in thread-local storage

**Internal Structures**:
```cpp
struct FTorchModelImpl {
    torch::jit::script::Module module;
    torch::Device device;
};

struct FTorchTensorImpl {
    torch::Tensor tensor;
};
```

### 3. C Header (`fortran_torch.h`)

**Purpose**: Define the C interface

**Key Features**:
- C-compatible types only
- Comprehensive documentation
- Consistent naming convention
- Clear error codes

## Data Flow

### Model Loading

```
Fortran                 C++                     LibTorch
──────────────────────────────────────────────────────────
torch_load_model()
    │
    └──> ftorch_load_model_c()
              │
              └──> torch::jit::load()
                        │
                        └──> Load TorchScript
                              │
                              ▼
              Create FTorchModelImpl
              │
              ▼
         Return c_ptr
         │
         ▼
    Store in torch_model%ptr
```

### Inference

```
Fortran                 C++                     LibTorch
──────────────────────────────────────────────────────────
torch_tensor_from_array()
    │
    └──> ftorch_create_tensor_c()
              │
              └──> torch::from_blob()
                        │
                        └──> Create tensor
                              │
                              ▼
              Create FTorchTensorImpl
              │
              ▼
         Return c_ptr

torch_forward(model, input)
    │
    └──> ftorch_forward_c()
              │
              └──> module.forward()
                        │
                        └──> Execute model
                              │
                              ▼
              Create output FTorchTensorImpl
              │
              ▼
         Return output c_ptr

torch_tensor_to_array()
    │
    └──> ftorch_tensor_to_array_c()
              │
              └──> tensor.to(CPU)
                        │
                        └──> memcpy to Fortran array
```

## Memory Management

### Ownership Model

- **Models**: Fortran code owns model handles, must call `torch_free_model`
- **Tensors**: Fortran code owns tensor handles, must call `torch_free_tensor`
- **Internal Data**: C++ wrappers manage LibTorch objects using RAII

### Memory Safety

```cpp
// RAII wrappers ensure cleanup
struct FTorchTensorImpl {
    torch::Tensor tensor;  // Automatically cleaned up when deleted
};

// Explicit cleanup from Fortran
void ftorch_free_tensor(FTorchTensor tensor) {
    if (tensor) {
        delete static_cast<FTorchTensorImpl*>(tensor);
    }
}
```

### Data Copying

**Tensor Creation**: Data is **copied** from Fortran array to PyTorch tensor
- Ensures Fortran array can be safely modified after
- Allows tensor to be moved to GPU if needed
- Small overhead, but necessary for safety

**Tensor Extraction**: Data is **copied** from PyTorch tensor to Fortran array
- Tensor may be on GPU, requires explicit copy to CPU
- Ensures contiguous layout for Fortran

## Error Handling

### Three-Level Strategy

1. **C++ Level**: Exceptions caught and converted to error codes
   ```cpp
   try {
       // PyTorch operations
   } catch (const c10::Error& e) {
       set_error(e.what());
       return FTORCH_ERROR_FORWARD_PASS;
   }
   ```

2. **C Level**: Status codes returned
   ```c
   FTorchStatus status = ftorch_forward(model, input, &output);
   ```

3. **Fortran Level**: Status checked, messages printed
   ```fortran
   if (status /= 0) then
       call print_last_error()
   end if
   ```

### Error Codes

```c
typedef enum {
    FTORCH_SUCCESS = 0,
    FTORCH_ERROR_NULL_POINTER = -1,
    FTORCH_ERROR_LOAD_MODEL = -2,
    FTORCH_ERROR_INVALID_TENSOR = -3,
    FTORCH_ERROR_FORWARD_PASS = -4,
    // ...
} FTorchStatus;
```

## Thread Safety

### Current Implementation

- **Models**: Not thread-safe by default
  - PyTorch models use global state
  - Recommend one model per thread if using OpenMP

- **Tensors**: Thread-safe for independent operations
  - Each tensor is independent
  - Safe to create/destroy in different threads

- **Error Messages**: Thread-local storage
  - Each thread has its own error message buffer
  - Safe for parallel execution

### Best Practices

```fortran
!$OMP PARALLEL
    type(torch_model) :: thread_model

    ! Each thread loads its own model copy
    thread_model = torch_load_model('model.pt')

    !$OMP DO
    do i = 1, n
        ! Safe: each thread has own model
        output = torch_forward(thread_model, input)
    end do
    !$OMP END DO

    call torch_free_model(thread_model)
!$OMP END PARALLEL
```

## Performance Considerations

### Bottlenecks

1. **Data Transfer**: Copying between Fortran and PyTorch
   - Minimized by batching
   - GPU transfer is expensive

2. **Model Loading**: One-time cost
   - Load once, reuse many times

3. **Inference**: Dominant cost for large models
   - Consider GPU for large models
   - Profile to understand actual cost

### Optimization Strategies

1. **Batching**: Process multiple inputs together
   ```fortran
   ! Instead of 100 calls with size (1, 10)
   ! Do 1 call with size (100, 10)
   ```

2. **Tensor Reuse**: Minimize allocations
   ```fortran
   ! Reuse tensors when possible
   do timestep = 1, n_timesteps
       input_tensor = torch_tensor_from_array(data)
       output_tensor = torch_forward(model, input_tensor)
       call torch_free_tensor(input_tensor)
       call torch_free_tensor(output_tensor)
   end do
   ```

3. **GPU Offload**: For large models
   ```fortran
   model = torch_load_model('model.pt', TORCH_DEVICE_CUDA)
   tensor = torch_tensor_from_array(data, TORCH_DEVICE_CUDA)
   ```

## Design Decisions

### Why C++ Wrapper?

**Alternative**: Direct Fortran to C++ binding
**Chosen**: C interface layer

**Rationale**:
- C has stable ABI (Fortran can easily call C)
- C++ exceptions don't cross language boundaries well
- Easier to handle name mangling
- More portable across Fortran compilers

### Why TorchScript?

**Alternative**: Python runtime, ONNX
**Chosen**: TorchScript (PyTorch C++ API)

**Rationale**:
- No Python runtime dependency
- Native PyTorch support
- Easy export from Python
- Good performance
- GPU support included

### Why Copy Data?

**Alternative**: Zero-copy, shared memory
**Chosen**: Explicit copying

**Rationale**:
- Safety: Avoids lifetime issues
- Simplicity: Clear ownership
- Portability: Works with any memory layout
- GPU: Necessary for GPU transfer anyway

### Why Generic Interfaces?

**Alternative**: Separate functions for each type/rank
**Chosen**: Fortran generic interfaces

**Rationale**:
- Better user experience
- Type safety at compile time
- Easier to extend
- Follows Fortran conventions

## Future Enhancements

### Planned

1. **Multiple Outputs**: Support models with multiple outputs
2. **In-place Operations**: Avoid some copies
3. **Async Execution**: Non-blocking inference
4. **Model Caching**: Share models between processes
5. **MPI Support**: Distributed inference

### Under Consideration

1. **Auto-batching**: Automatic batch collection
2. **Tensor Views**: Zero-copy when possible
3. **Custom Operators**: Register Fortran functions
4. **Quantization**: Int8 inference support

## Testing Strategy

### Unit Tests

- Test each function independently
- Cover error cases
- Check memory management

### Integration Tests

- End-to-end workflows
- Multiple models
- Different data types

### Performance Tests

- Benchmark inference time
- Memory usage profiling
- Scaling tests

## Build System

### CMake Design

```
CMakeLists.txt (root)
├── Find LibTorch
├── Build C++ library
├── Build Fortran module
└── Build examples

examples/fortran/CMakeLists.txt
└── Link executables
```

### Dependencies

```
Fortran-Torch
├── Depends on: LibTorch (required)
├── Depends on: C++ compiler (build)
├── Depends on: Fortran compiler (build)
└── Depends on: CMake (build)
```

## Deployment

### Library Distribution

Fortran-Torch produces:
- `libfortran_torch_cpp.so` - C++ wrapper
- `libftorch.so` - Fortran module
- `fortran_torch.h` - C header
- `ftorch.mod` - Fortran module file

### User Application

Links against:
- Fortran-Torch libraries
- LibTorch libraries

```cmake
target_link_libraries(my_app
    ftorch
    fortran_torch_cpp
    ${TORCH_LIBRARIES}
)
```

## Documentation

### Levels

1. **User Guide**: README.md - How to use
2. **Installation**: INSTALL.md - How to build
3. **API Reference**: In-code comments
4. **Architecture**: This document - How it works
5. **Contributing**: CONTRIBUTING.md - How to contribute

---

For questions about architecture or design decisions, please open a GitHub issue.
