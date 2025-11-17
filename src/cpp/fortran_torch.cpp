#include "fortran_torch.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <cstring>
#include <vector>
#include <exception>

// Thread-local storage for error messages
static thread_local std::string last_error_message;

// Helper function to set error message
static void set_error(const std::string& msg) {
    last_error_message = msg;
}

// Convert FTorchDevice to torch::Device
static torch::Device get_device(FTorchDevice device) {
    switch (device) {
        case FTORCH_DEVICE_CUDA:
            return torch::kCUDA;
        case FTORCH_DEVICE_CPU:
        default:
            return torch::kCPU;
    }
}

// Convert FTorchDType to torch::ScalarType
static torch::ScalarType get_dtype(FTorchDType dtype) {
    switch (dtype) {
        case FTORCH_FLOAT32:
            return torch::kFloat32;
        case FTORCH_FLOAT64:
            return torch::kFloat64;
        case FTORCH_INT32:
            return torch::kInt32;
        case FTORCH_INT64:
            return torch::kInt64;
        default:
            return torch::kFloat32;
    }
}

// Wrapper struct for torch::jit::script::Module
struct FTorchModelImpl {
    torch::jit::script::Module module;
    torch::Device device;

    FTorchModelImpl(torch::Device dev) : device(dev) {}
};

// Wrapper struct for torch::Tensor
struct FTorchTensorImpl {
    torch::Tensor tensor;
};

extern "C" {

FTorchStatus ftorch_load_model(const char* model_path, FTorchDevice device, FTorchModel* model) {
    if (!model_path || !model) {
        set_error("Null pointer provided to ftorch_load_model");
        return FTORCH_ERROR_NULL_POINTER;
    }

    try {
        torch::Device torch_device = get_device(device);

        // Create model wrapper
        FTorchModelImpl* impl = new FTorchModelImpl(torch_device);

        // Load the model
        impl->module = torch::jit::load(model_path, torch_device);
        impl->module.eval();  // Set to evaluation mode

        *model = static_cast<FTorchModel>(impl);
        set_error("");
        return FTORCH_SUCCESS;

    } catch (const c10::Error& e) {
        set_error(std::string("PyTorch error: ") + e.what());
        return FTORCH_ERROR_LOAD_MODEL;
    } catch (const std::exception& e) {
        set_error(std::string("Error loading model: ") + e.what());
        return FTORCH_ERROR_LOAD_MODEL;
    }
}

void ftorch_free_model(FTorchModel model) {
    if (model) {
        FTorchModelImpl* impl = static_cast<FTorchModelImpl*>(model);
        delete impl;
    }
}

FTorchStatus ftorch_create_tensor(const void* data, int ndim, const int64_t* shape,
                                   FTorchDType dtype, FTorchDevice device,
                                   FTorchTensor* tensor) {
    if (!data || !shape || !tensor) {
        set_error("Null pointer provided to ftorch_create_tensor");
        return FTORCH_ERROR_NULL_POINTER;
    }

    try {
        // Create shape vector
        std::vector<int64_t> sizes(shape, shape + ndim);

        torch::ScalarType torch_dtype = get_dtype(dtype);
        torch::Device torch_device = get_device(device);

        // Create tensor from data
        torch::TensorOptions options = torch::TensorOptions()
            .dtype(torch_dtype)
            .device(torch::kCPU);  // Always create on CPU first

        torch::Tensor t;

        // Calculate total number of elements
        int64_t numel = 1;
        for (int i = 0; i < ndim; ++i) {
            numel *= sizes[i];
        }

        // Create tensor based on data type
        switch (dtype) {
            case FTORCH_FLOAT32: {
                const float* float_data = static_cast<const float*>(data);
                t = torch::from_blob(const_cast<float*>(float_data), sizes, options).clone();
                break;
            }
            case FTORCH_FLOAT64: {
                const double* double_data = static_cast<const double*>(data);
                t = torch::from_blob(const_cast<double*>(double_data), sizes, options).clone();
                break;
            }
            case FTORCH_INT32: {
                const int32_t* int_data = static_cast<const int32_t*>(data);
                t = torch::from_blob(const_cast<int32_t*>(int_data), sizes, options).clone();
                break;
            }
            case FTORCH_INT64: {
                const int64_t* long_data = static_cast<const int64_t*>(data);
                t = torch::from_blob(const_cast<int64_t*>(long_data), sizes, options).clone();
                break;
            }
            default:
                set_error("Unsupported data type");
                return FTORCH_ERROR_INVALID_TENSOR;
        }

        // Move to target device if necessary
        if (device != FTORCH_DEVICE_CPU) {
            t = t.to(torch_device);
        }

        FTorchTensorImpl* impl = new FTorchTensorImpl{t};
        *tensor = static_cast<FTorchTensor>(impl);

        set_error("");
        return FTORCH_SUCCESS;

    } catch (const c10::Error& e) {
        set_error(std::string("PyTorch error: ") + e.what());
        return FTORCH_ERROR_INVALID_TENSOR;
    } catch (const std::exception& e) {
        set_error(std::string("Error creating tensor: ") + e.what());
        return FTORCH_ERROR_INVALID_TENSOR;
    }
}

FTorchStatus ftorch_tensor_to_array(FTorchTensor tensor, void* data) {
    if (!tensor || !data) {
        set_error("Null pointer provided to ftorch_tensor_to_array");
        return FTORCH_ERROR_NULL_POINTER;
    }

    try {
        FTorchTensorImpl* impl = static_cast<FTorchTensorImpl*>(tensor);

        // Move tensor to CPU if necessary
        torch::Tensor cpu_tensor = impl->tensor.to(torch::kCPU).contiguous();

        // Copy data
        size_t byte_size = cpu_tensor.numel() * cpu_tensor.element_size();
        std::memcpy(data, cpu_tensor.data_ptr(), byte_size);

        set_error("");
        return FTORCH_SUCCESS;

    } catch (const c10::Error& e) {
        set_error(std::string("PyTorch error: ") + e.what());
        return FTORCH_ERROR_INVALID_TENSOR;
    } catch (const std::exception& e) {
        set_error(std::string("Error copying tensor data: ") + e.what());
        return FTORCH_ERROR_INVALID_TENSOR;
    }
}

FTorchStatus ftorch_tensor_shape(FTorchTensor tensor, int* ndim, int64_t* shape) {
    if (!tensor || !ndim || !shape) {
        set_error("Null pointer provided to ftorch_tensor_shape");
        return FTORCH_ERROR_NULL_POINTER;
    }

    try {
        FTorchTensorImpl* impl = static_cast<FTorchTensorImpl*>(tensor);

        *ndim = impl->tensor.dim();
        auto sizes = impl->tensor.sizes();

        for (int i = 0; i < *ndim; ++i) {
            shape[i] = sizes[i];
        }

        set_error("");
        return FTORCH_SUCCESS;

    } catch (const std::exception& e) {
        set_error(std::string("Error getting tensor shape: ") + e.what());
        return FTORCH_ERROR_INVALID_TENSOR;
    }
}

void ftorch_free_tensor(FTorchTensor tensor) {
    if (tensor) {
        FTorchTensorImpl* impl = static_cast<FTorchTensorImpl*>(tensor);
        delete impl;
    }
}

FTorchStatus ftorch_forward(FTorchModel model, FTorchTensor input, FTorchTensor* output) {
    if (!model || !input || !output) {
        set_error("Null pointer provided to ftorch_forward");
        return FTORCH_ERROR_NULL_POINTER;
    }

    try {
        FTorchModelImpl* model_impl = static_cast<FTorchModelImpl*>(model);
        FTorchTensorImpl* input_impl = static_cast<FTorchTensorImpl*>(input);

        // Run forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_impl->tensor);

        torch::jit::IValue output_ivalue = model_impl->module.forward(inputs);

        // Extract output tensor
        torch::Tensor output_tensor = output_ivalue.toTensor();

        FTorchTensorImpl* output_impl = new FTorchTensorImpl{output_tensor};
        *output = static_cast<FTorchTensor>(output_impl);

        set_error("");
        return FTORCH_SUCCESS;

    } catch (const c10::Error& e) {
        set_error(std::string("PyTorch error during forward pass: ") + e.what());
        return FTORCH_ERROR_FORWARD_PASS;
    } catch (const std::exception& e) {
        set_error(std::string("Error during forward pass: ") + e.what());
        return FTORCH_ERROR_FORWARD_PASS;
    }
}

FTorchStatus ftorch_forward_multi(FTorchModel model, FTorchTensor* inputs,
                                   int n_inputs, FTorchTensor* output) {
    if (!model || !inputs || !output || n_inputs <= 0) {
        set_error("Invalid parameters provided to ftorch_forward_multi");
        return FTORCH_ERROR_NULL_POINTER;
    }

    try {
        FTorchModelImpl* model_impl = static_cast<FTorchModelImpl*>(model);

        // Prepare input tensors
        std::vector<torch::jit::IValue> torch_inputs;
        for (int i = 0; i < n_inputs; ++i) {
            FTorchTensorImpl* input_impl = static_cast<FTorchTensorImpl*>(inputs[i]);
            torch_inputs.push_back(input_impl->tensor);
        }

        // Run forward pass
        torch::jit::IValue output_ivalue = model_impl->module.forward(torch_inputs);

        // Extract output tensor
        torch::Tensor output_tensor = output_ivalue.toTensor();

        FTorchTensorImpl* output_impl = new FTorchTensorImpl{output_tensor};
        *output = static_cast<FTorchTensor>(output_impl);

        set_error("");
        return FTORCH_SUCCESS;

    } catch (const c10::Error& e) {
        set_error(std::string("PyTorch error during forward pass: ") + e.what());
        return FTORCH_ERROR_FORWARD_PASS;
    } catch (const std::exception& e) {
        set_error(std::string("Error during forward pass: ") + e.what());
        return FTORCH_ERROR_FORWARD_PASS;
    }
}

int ftorch_cuda_available(void) {
    return torch::cuda::is_available() ? 1 : 0;
}

const char* ftorch_get_last_error(void) {
    return last_error_message.c_str();
}

} // extern "C"
