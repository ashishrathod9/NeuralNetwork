#ifndef LAYER_H
#define LAYER_H

#include "Tensor.h"

namespace nn {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update_parameters(float learning_rate) = 0;
    virtual std::vector<Tensor*> get_parameters() = 0;  // Get parameters for optimizers
    virtual std::vector<Tensor*> get_gradients() = 0;   // Get gradients for optimizers
};

class Linear : public Layer {
public:
    Linear(size_t input_size, size_t output_size);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(float learning_rate) override;
    std::vector<Tensor*> get_parameters() override;
    std::vector<Tensor*> get_gradients() override;
    
    void set_weights(const Tensor& weights);
    void set_bias(const Tensor& bias);
    
private:
    Tensor weights_;
    Tensor bias_;
    Tensor input_cache_;  // Store input for backward pass
    
    // Gradients
    Tensor grad_weights_;
    Tensor grad_bias_;
};

class Sigmoid : public Layer {
public:
    Sigmoid() = default;
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(float learning_rate) override {}
    std::vector<Tensor*> get_parameters() override { return {}; }
    std::vector<Tensor*> get_gradients() override { return {}; }
    
private:
    Tensor output_cache_;  // Store output for backward pass
};

class ReLU : public Layer {
public:
    ReLU() = default;
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(float learning_rate) override {}
    std::vector<Tensor*> get_parameters() override { return {}; }
    std::vector<Tensor*> get_gradients() override { return {}; }
    
private:
    Tensor input_cache_;  // Store input for backward pass
};

class Tanh : public Layer {
public:
    Tanh() = default;
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(float learning_rate) override {}
    std::vector<Tensor*> get_parameters() override { return {}; }
    std::vector<Tensor*> get_gradients() override { return {}; }
    
private:
    Tensor output_cache_;  // Store output for backward pass
};

} // namespace nn

#endif // LAYER_H