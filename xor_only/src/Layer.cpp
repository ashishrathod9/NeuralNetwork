#include "Layer.h"
#include <random>

namespace nn {

Linear::Linear(size_t input_size, size_t output_size) 
    : weights_(output_size, input_size), bias_(output_size, 1), 
      grad_weights_(output_size, input_size), grad_bias_(output_size, 1) {
    // Initialize weights randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = dis(gen);
    }
    
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] = dis(gen);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // Store input for backward pass
    input_cache_ = input;
    
    // Compute: output = weights * input + bias
    Tensor weighted_sum = weights_.matmul(input);
    
    // Add bias (broadcasting: bias is (output_size, 1), weighted_sum is (output_size, batch_size))
    Tensor output(weighted_sum.rows(), weighted_sum.cols());
    for (size_t i = 0; i < weighted_sum.rows(); ++i) {
        for (size_t j = 0; j < weighted_sum.cols(); ++j) {
            output(i, j) = weighted_sum(i, j) + bias_(i, 0);
        }
    }
    
    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // Compute gradients
    // grad_weights = grad_output * input^T
    grad_weights_ = grad_output.matmul(input_cache_.transpose());
    
    // grad_bias = sum(grad_output, axis=1) (sum along batch dimension)
    grad_bias_ = grad_output.sum(1);  // Sum along columns to get (output_size, 1)
    
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
    if (weights.shape() == weights_.shape()) {
        weights_ = weights;
    }
}

void Linear::set_bias(const Tensor& bias) {
    if (bias.shape() == bias_.shape()) {
        bias_ = bias;
    }
}

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor output = input.sigmoid();
    output_cache_ = output;  // Store for backward pass
    return output;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
    Tensor one_tensor(output_cache_.rows(), output_cache_.cols());
    one_tensor.fill(1.0f);
    Tensor sig_derivative = output_cache_ * (one_tensor - output_cache_);  // sig * (1 - sig)
    return grad_output * sig_derivative;
}

} // namespace nn