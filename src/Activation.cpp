#include "Layer.h"
#include <algorithm>

namespace nn {

// Sigmoid activation layer
Tensor Sigmoid::forward(const Tensor& input) {
    output_cache_ = input.sigmoid();
    return output_cache_;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
    Tensor grad_input(output_cache_.rows(), output_cache_.cols());
    
    for (size_t i = 0; i < output_cache_.size(); ++i) {
        float sig = output_cache_[i];
        grad_input[i] = grad_output[i] * sig * (1.0f - sig);
    }
    
    return grad_input;
}

// ReLU activation layer
Tensor ReLU::forward(const Tensor& input) {
    input_cache_ = input;  // Store input for backward pass
    return input.relu();
}

Tensor ReLU::backward(const Tensor& grad_output) {
    // Derivative of ReLU: 1 if x > 0, 0 otherwise
    Tensor grad_input(input_cache_.rows(), input_cache_.cols());
    
    for (size_t i = 0; i < input_cache_.size(); ++i) {
        grad_input[i] = input_cache_[i] > 0.0f ? grad_output[i] : 0.0f;
    }
    
    return grad_input;
}

// Tanh activation layer
Tensor Tanh::forward(const Tensor& input) {
    output_cache_ = input.tanh_activation();
    return output_cache_;
}

Tensor Tanh::backward(const Tensor& grad_output) {
    // Derivative of tanh: 1 - tanh(x)^2
    Tensor grad_input(output_cache_.rows(), output_cache_.cols());
    
    for (size_t i = 0; i < output_cache_.size(); ++i) {
        float tanh_val = output_cache_[i];
        grad_input[i] = grad_output[i] * (1.0f - tanh_val * tanh_val);
    }
    
    return grad_input;
}

} // namespace nn