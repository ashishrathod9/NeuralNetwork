#include "Layer.h"
#include <random>

namespace nn {

Linear::Linear(size_t input_size, size_t output_size) 
    : weights_(output_size, input_size), bias_(output_size, 1), 
      grad_weights_(output_size, input_size), grad_bias_(output_size, 1) {
    
    // Initialize weights with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.1f);
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = dis(gen);
    }
    
    // Initialize bias to zeros
    bias_.fill(0.0f);
}

Tensor Linear::forward(const Tensor& input) {
    // Store input for backward pass
    input_cache_ = input;
    
    // Compute output = weights * input + bias
    Tensor output = weights_.matmul(input);
    
    // Add bias (broadcasting)
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += bias_(i, 0);
        }
    }
    
    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // grad_weights = grad_output * input^T
    grad_weights_ = grad_output.matmul(input_cache_.transpose());
    
    // grad_bias = grad_output (sum along batch dimension if needed)
    grad_bias_ = grad_output.sum(1);  // Sum along the column dimension (batch dimension)
    
    // grad_input = weights^T * grad_output
    Tensor grad_input = weights_.transpose().matmul(grad_output);
    
    return grad_input;
}

void Linear::update_parameters(float learning_rate) {
    // Update weights: weights = weights - learning_rate * grad_weights
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_weights_[i];
    }
    
    // Update bias: bias = bias - learning_rate * grad_bias
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate * grad_bias_[i];
    }
}

std::vector<Tensor*> Linear::get_parameters() {
    return {&weights_, &bias_};
}

std::vector<Tensor*> Linear::get_gradients() {
    return {&grad_weights_, &grad_bias_};
}

void Linear::set_weights(const Tensor& weights) {
    if (weights.shape() != weights_.shape()) {
        throw std::runtime_error("Weight tensor shape mismatch");
    }
    weights_ = weights;
}

void Linear::set_bias(const Tensor& bias) {
    if (bias.shape() != bias_.shape()) {
        throw std::runtime_error("Bias tensor shape mismatch");
    }
    bias_ = bias;
}

} // namespace nn