#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"
#include <vector>
#include <map>

namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<Layer*>& layers) = 0;
    virtual void zero_grad(std::vector<Layer*>& layers) = 0;
};

class SGD : public Optimizer {
public:
    SGD(float learning_rate = 0.01f);
    
    void step(std::vector<Layer*>& layers) override;
    void zero_grad(std::vector<Layer*>& layers) override;
    
private:
    float learning_rate_;
};

class Momentum : public Optimizer {
public:
    Momentum(float learning_rate = 0.01f, float momentum = 0.9f);
    
    void step(std::vector<Layer*>& layers) override;
    void zero_grad(std::vector<Layer*>& layers) override;
    
private:
    float learning_rate_;
    float momentum_;
    std::map<Layer*, std::vector<float>> velocities_;  // Track velocity for each layer's parameters
};

class Adam : public Optimizer {
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    
    void step(std::vector<Layer*>& layers) override;
    void zero_grad(std::vector<Layer*>& layers) override;
    
private:
    float learning_rate_;
    float beta1_, beta2_;
    float epsilon_;
    int timestep_;
    
    struct AdamParams {
        std::vector<float> m_w, v_w;  // First and second moment estimates for weights
        std::vector<float> m_b, v_b;  // First and second moment estimates for biases
        size_t w_size, b_size;
    };
    
    std::map<Layer*, AdamParams> params_;
};

} // namespace nn

#endif // OPTIMIZER_H