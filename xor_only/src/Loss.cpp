#include "Loss.h"

namespace nn {

float MSELoss::compute(const Tensor& predicted, const Tensor& actual) {
    if (predicted.shape() != actual.shape()) {
        throw std::runtime_error("Predicted and actual tensor shapes do not match");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        float diff = predicted[i] - actual[i];
        sum += diff * diff;
    }
    
    return sum / predicted.size();
}

Tensor MSELoss::compute_gradient(const Tensor& predicted, const Tensor& actual) {
    if (predicted.shape() != actual.shape()) {
        throw std::runtime_error("Predicted and actual tensor shapes do not match");
    }
    
    // For MSE: d/dx [(x - t)^2] = 2 * (x - t) / n
    size_t n = predicted.size();
    Tensor gradient(predicted.shape()[0], predicted.shape()[1]);
    
    for (size_t i = 0; i < predicted.size(); ++i) {
        gradient[i] = 2.0f * (predicted[i] - actual[i]) / n;
    }
    
    return gradient;
}

} // namespace nn