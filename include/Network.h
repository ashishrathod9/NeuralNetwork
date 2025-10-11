#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include <vector>

namespace nn {

class Network {
public:
    Network();
    
    void add_layer(Layer* layer);
    Tensor forward(const Tensor& input);
    float train(const Tensor& input, const Tensor& target, Loss& loss, Optimizer& optimizer);
    float evaluate(const Tensor& input, const Tensor& target, Loss& loss);
    
    std::vector<Layer*>& get_layers() { return layers_; }
    
private:
    std::vector<Layer*> layers_;
    std::vector<Tensor> layer_outputs_;  // Cache outputs for backward pass
};

} // namespace nn

#endif // NETWORK_H