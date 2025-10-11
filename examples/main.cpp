#include "Network.h"
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Neural Network Library - Basic Example" << std::endl;
    
    // Create a simple network: 2 -> 4 -> 1
    nn::Network net;
    
    // Add layers
    net.add_layer(new nn::Linear(2, 4));  // Input: 2 features, Hidden: 4 neurons
    net.add_layer(new nn::Linear(4, 1));  // Hidden: 4 neurons, Output: 1 neuron
    
    // Create loss function and optimizer
    nn::MSELoss loss_fn;
    nn::SGD optimizer(0.1f);  // Learning rate = 0.1
    
    // Simple XOR-like training data
    std::vector<std::vector<float>> inputs_data = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    std::vector<std::vector<float>> targets_data = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };
    
    // Convert to tensors
    std::vector<nn::Tensor> inputs, targets;
    for (const auto& input : inputs_data) {
        inputs.push_back(nn::Tensor(input, {input.size(), 1}));
    }
    for (const auto& target : targets_data) {
        targets.push_back(nn::Tensor(target, {target.size(), 1}));
    }
    
    // Training loop
    std::cout << "Training the network..." << std::endl;
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            float loss = net.train(inputs[i], targets[i], loss_fn, optimizer);
            total_loss += loss;
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Average Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
    
    // Test the trained network
    std::cout << "\nTesting the trained network:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        nn::Tensor output = net.forward(inputs[i]);
        std::cout << "Input: [" << inputs_data[i][0] << ", " << inputs_data[i][1] 
                  << "] -> Output: " << output(0, 0) << ", Target: " << targets_data[i][0] << std::endl;
    }
    
    // Clean up allocated memory
    auto& layers = net.get_layers();
    for (auto* layer : layers) {
        delete layer;
    }
    
    return 0;
}