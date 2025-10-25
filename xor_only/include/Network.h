#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include <vector>

namespace nn {

class Network {
public:
    Network();
    ~Network(); // Added destructor for memory cleanup
    
    void add_layer(Layer* layer);
    Tensor forward(const Tensor& input);
    void train_step(const Tensor& input, const Tensor& target, float learning_rate);
    
    std::vector<Layer*>& get_layers() { return layers_; }
    const std::vector<Layer*>& get_layers() const { return layers_; }
    
private:
    std::vector<Layer*> layers_;
    std::vector<Tensor> layer_outputs_;  // Cache outputs for backward pass
};

} // namespace nn

#endif // NETWORK_H