#include "Optimizer.h"
#include <numeric>

namespace nn {

SGD::SGD(float learning_rate) : learning_rate_(learning_rate) {}

void SGD::step(std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        auto params = layer->get_parameters();
        auto grads = layer->get_gradients();
        
        if (params.size() != grads.size()) {
            continue;  // Skip layers without matching parameters and gradients
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] && grads[i]) {
                // Update parameters: param = param - learning_rate * grad
                for (size_t j = 0; j < params[i]->size(); ++j) {
                    (*params[i])[j] -= learning_rate_ * (*grads[i])[j];
                }
            }
        }
    }
}

void SGD::zero_grad(std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        auto grads = layer->get_gradients();
        for (auto* grad : grads) {
            if (grad) {
                for (size_t i = 0; i < grad->size(); ++i) {
                    (*grad)[i] = 0.0f;
                }
            }
        }
    }
}

Momentum::Momentum(float learning_rate, float momentum) 
    : learning_rate_(learning_rate), momentum_(momentum) {}

void Momentum::step(std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        auto params = layer->get_parameters();
        auto grads = layer->get_gradients();
        
        if (params.size() != grads.size()) {
            continue;  // Skip layers without matching parameters and gradients
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            if (!params[i] || !grads[i]) continue;
            
            // Initialize velocity for this parameter if it hasn't been
            if (velocities_.find(layer) == velocities_.end()) {
                velocities_[layer] = std::vector<float>(params[i]->size(), 0.0f);
            }
            
            auto& velocity = velocities_[layer];
            if (velocity.size() != params[i]->size()) {
                velocity.resize(params[i]->size(), 0.0f);
            }
            
            // Update velocity: v = momentum * v - learning_rate * grad
            // Update parameters: param = param + v
            for (size_t j = 0; j < params[i]->size(); ++j) {
                velocity[j] = momentum_ * velocity[j] - learning_rate_ * (*grads[i])[j];
                (*params[i])[j] += velocity[j];
            }
        }
    }
}

void Momentum::zero_grad(std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        auto grads = layer->get_gradients();
        for (auto* grad : grads) {
            if (grad) {
                for (size_t i = 0; i < grad->size(); ++i) {
                    (*grad)[i] = 0.0f;
                }
            }
        }
    }
}

Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), timestep_(0) {}

void Adam::step(std::vector<Layer*>& layers) {
    timestep_++;
    
    float lr_t = learning_rate_ * std::sqrt(1 - std::pow(beta2_, timestep_)) / (1 - std::pow(beta1_, timestep_));
    
    for (Layer* layer : layers) {
        auto params = layer->get_parameters();
        auto grads = layer->get_gradients();
        
        if (params.size() != grads.size()) {
            continue;  // Skip layers without matching parameters and gradients
        }
        
        // Initialize Adam parameters for this layer if they don't exist
        if (params_.find(layer) == params_.end()) {
            params_[layer] = AdamParams{};
            auto& adam_params = params_[layer];
            
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]) {
                    if (i == 0) {  // Assume first parameter is weights
                        adam_params.w_size = params[i]->size();
                        adam_params.m_w.resize(adam_params.w_size, 0.0f);
                        adam_params.v_w.resize(adam_params.w_size, 0.0f);
                    } else {  // Assume second parameter is bias
                        adam_params.b_size = params[i]->size();
                        if (adam_params.m_b.size() != adam_params.b_size) {
                            adam_params.m_b.resize(adam_params.b_size, 0.0f);
                            adam_params.v_b.resize(adam_params.b_size, 0.0f);
                        }
                    }
                }
            }
        }
        
        auto& adam_params = params_[layer];
        
        // Update weights if they exist
        if (params.size() > 0 && params[0] && grads[0]) {
            for (size_t j = 0; j < params[0]->size(); ++j) {
                adam_params.m_w[j] = beta1_ * adam_params.m_w[j] + (1 - beta1_) * (*grads[0])[j];
                adam_params.v_w[j] = beta2_ * adam_params.v_w[j] + (1 - beta2_) * (*grads[0])[j] * (*grads[0])[j];
                
                float m_hat = adam_params.m_w[j] / (1 - std::pow(beta1_, timestep_));
                float v_hat = adam_params.v_w[j] / (1 - std::pow(beta2_, timestep_));
                
                (*params[0])[j] -= lr_t * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
        
        // Update bias if it exists
        if (params.size() > 1 && params[1] && grads[1]) {
            for (size_t j = 0; j < params[1]->size(); ++j) {
                adam_params.m_b[j] = beta1_ * adam_params.m_b[j] + (1 - beta1_) * (*grads[1])[j];
                adam_params.v_b[j] = beta2_ * adam_params.v_b[j] + (1 - beta2_) * (*grads[1])[j] * (*grads[1])[j];
                
                float m_hat = adam_params.m_b[j] / (1 - std::pow(beta1_, timestep_));
                float v_hat = adam_params.v_b[j] / (1 - std::pow(beta2_, timestep_));
                
                (*params[1])[j] -= lr_t * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
}

void Adam::zero_grad(std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        auto grads = layer->get_gradients();
        for (auto* grad : grads) {
            if (grad) {
                for (size_t i = 0; i < grad->size(); ++i) {
                    (*grad)[i] = 0.0f;
                }
            }
        }
    }
}

} // namespace nn