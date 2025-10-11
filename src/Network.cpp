#include "Network.h"

namespace nn {

Network::Network() {}

void Network::add_layer(Layer* layer) {
    layers_.push_back(layer);
}

Tensor Network::forward(const Tensor& input) {
    Tensor current_output = input;
    
    for (Layer* layer : layers_) {
        current_output = layer->forward(current_output);
    }
    
    return current_output;
}

float Network::train(const Tensor& input, const Tensor& target, Loss& loss, Optimizer& optimizer) {
    // Forward pass
    Tensor output = forward(input);
    
    // Compute loss
    float loss_value = loss.compute_loss(output, target);
    
    // Compute gradient of loss
    Tensor grad_output = loss.compute_gradient(output, target);
    
    // Backward pass
    // Propagate gradients backwards through layers
    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad_output = layers_[i]->backward(grad_output);
    }
    
    // Update parameters
    optimizer.step(layers_);
    
    return loss_value;
}

float Network::evaluate(const Tensor& input, const Tensor& target, Loss& loss) {
    Tensor output = forward(input);
    return loss.compute_loss(output, target);
}

} // namespace nn