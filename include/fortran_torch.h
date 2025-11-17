#ifndef FORTRAN_TORCH_H
#define FORTRAN_TORCH_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for PyTorch models
typedef void* FTorchModel;

// Opaque handle for tensors
typedef void* FTorchTensor;

// Error codes
typedef enum {
    FTORCH_SUCCESS = 0,
    FTORCH_ERROR_NULL_POINTER = -1,
    FTORCH_ERROR_LOAD_MODEL = -2,
    FTORCH_ERROR_INVALID_TENSOR = -3,
    FTORCH_ERROR_FORWARD_PASS = -4,
    FTORCH_ERROR_DIMENSION_MISMATCH = -5,
    FTORCH_ERROR_UNKNOWN = -99
} FTorchStatus;

// Data types matching PyTorch
typedef enum {
    FTORCH_FLOAT32 = 0,
    FTORCH_FLOAT64 = 1,
    FTORCH_INT32 = 2,
    FTORCH_INT64 = 3
} FTorchDType;

// Device types
typedef enum {
    FTORCH_DEVICE_CPU = 0,
    FTORCH_DEVICE_CUDA = 1
} FTorchDevice;

/**
 * Load a TorchScript model from file
 * @param model_path Path to the .pt model file
 * @param device Device to load model on (CPU or CUDA)
 * @param model Output model handle
 * @return Status code
 */
FTorchStatus ftorch_load_model(const char* model_path, FTorchDevice device, FTorchModel* model);

/**
 * Free a loaded model
 * @param model Model handle to free
 */
void ftorch_free_model(FTorchModel model);

/**
 * Create a tensor from raw data
 * @param data Pointer to data array
 * @param ndim Number of dimensions
 * @param shape Array of dimension sizes
 * @param dtype Data type
 * @param device Device to create tensor on
 * @param tensor Output tensor handle
 * @return Status code
 */
FTorchStatus ftorch_create_tensor(const void* data, int ndim, const int64_t* shape,
                                   FTorchDType dtype, FTorchDevice device,
                                   FTorchTensor* tensor);

/**
 * Get data from a tensor
 * @param tensor Tensor handle
 * @param data Output buffer (must be pre-allocated)
 * @return Status code
 */
FTorchStatus ftorch_tensor_to_array(FTorchTensor tensor, void* data);

/**
 * Get tensor shape
 * @param tensor Tensor handle
 * @param ndim Output: number of dimensions
 * @param shape Output: array of dimension sizes (must be pre-allocated)
 * @return Status code
 */
FTorchStatus ftorch_tensor_shape(FTorchTensor tensor, int* ndim, int64_t* shape);

/**
 * Free a tensor
 * @param tensor Tensor handle to free
 */
void ftorch_free_tensor(FTorchTensor tensor);

/**
 * Run forward pass on model with single input
 * @param model Model handle
 * @param input Input tensor
 * @param output Output tensor handle
 * @return Status code
 */
FTorchStatus ftorch_forward(FTorchModel model, FTorchTensor input, FTorchTensor* output);

/**
 * Run forward pass on model with multiple inputs
 * @param model Model handle
 * @param inputs Array of input tensors
 * @param n_inputs Number of input tensors
 * @param output Output tensor handle
 * @return Status code
 */
FTorchStatus ftorch_forward_multi(FTorchModel model, FTorchTensor* inputs,
                                   int n_inputs, FTorchTensor* output);

/**
 * Check if CUDA is available
 * @return 1 if CUDA is available, 0 otherwise
 */
int ftorch_cuda_available(void);

/**
 * Get error message for last error
 * @return Error message string
 */
const char* ftorch_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // FORTRAN_TORCH_H
