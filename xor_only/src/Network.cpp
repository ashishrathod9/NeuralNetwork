#include "Network.h"
#include <iostream>

namespace nn {

Network::Network() {}

Network::~Network() {
    // Clean up allocated memory
    for (auto* layer : layers_) {
        delete layer;
    }
}

void Network::add_layer(Layer* layer) {
    layers_.push_back(layer);
}

Tensor Network::forward(const Tensor& input) {
    Tensor current_input = input;
    
    for (auto* layer : layers_) {
        current_input = layer->forward(current_input);
    }
    
    return current_input;
}

void Network::train_step(const Tensor& input, const Tensor& target, float learning_rate) {
    // Forward pass - store outputs for backward pass
    layer_outputs_.clear();
    Tensor current_input = input;
    layer_outputs_.push_back(current_input);
    
    for (auto* layer : layers_) {
        current_input = layer->forward(current_input);
        layer_outputs_.push_back(current_input);
    }
    
    // Compute initial gradient (derivative of loss w.r.t. output)
    // For MSE: d/dx [(x - t)^2] = 2 * (x - t)
    Tensor grad_output = (current_input - target) * 2.0f;
    
    // Backward pass - propagate gradients through layers in reverse order
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad_output = layers_[i]->backward(grad_output);
        layers_[i]->update_parameters(learning_rate);
    }
}

} // namespace nn