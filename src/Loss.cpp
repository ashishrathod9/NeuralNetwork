#include "Loss.h"
#include <cmath>

namespace nn {

// MSE Loss Implementation
float MSELoss::compute_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Prediction and target shapes do not match");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    
    return sum / static_cast<float>(predictions.size());
}

Tensor MSELoss::compute_gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Prediction and target shapes do not match");
    }
    
    Tensor gradient(predictions.rows(), predictions.cols());
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        gradient[i] = 2.0f * (predictions[i] - targets[i]) / static_cast<float>(predictions.size());
    }
    
    return gradient;
}

// Cross Entropy Loss Implementation
float CrossEntropyLoss::compute_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Prediction and target shapes do not match");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        // Add small epsilon to prevent log(0)
        float p = std::max(predictions[i], 1e-15f);
        sum -= targets[i] * std::log(p);
    }
    
    return sum / static_cast<float>(predictions.size());
}

Tensor CrossEntropyLoss::compute_gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Prediction and target shapes do not match");
    }
    
    Tensor gradient(predictions.rows(), predictions.cols());
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        // Add small epsilon to prevent division by zero
        float p = std::max(predictions[i], 1e-15f);
        gradient[i] = (p - targets[i]) / (p * static_cast<float>(predictions.size()));
    }
    
    return gradient;
}

} // namespace nn