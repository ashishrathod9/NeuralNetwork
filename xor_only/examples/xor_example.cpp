#include "Network.h"
#include "Layer.h"
#include "Loss.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Neural Network Library - XOR Example" << std::endl;
    std::cout << "Training a network to solve the XOR problem" << std::endl;
    
    // Create a network: 2 -> 4 -> 4 -> 1
    nn::Network net;
    
    // Add layers: Input(2) -> Hidden(4) -> Hidden(4) -> Output(1)
    net.add_layer(new nn::Linear(2, 4));
    net.add_layer(new nn::Sigmoid());  // Activation after first linear layer
    net.add_layer(new nn::Linear(4, 4));
    net.add_layer(new nn::Sigmoid());  // Activation after second linear layer
    net.add_layer(new nn::Linear(4, 1));
    net.add_layer(new nn::Sigmoid());  // Final activation
    
    // Prepare XOR training data
    std::vector<nn::Tensor> train_data = {
        nn::Tensor({{0.0f, 0.0f}}, {2, 1}),
        nn::Tensor({{0.0f, 1.0f}}, {2, 1}),
        nn::Tensor({{1.0f, 0.0f}}, {2, 1}),
        nn::Tensor({{1.0f, 1.0f}}, {2, 1})
    };
    
    std::vector<nn::Tensor> train_targets = {
        nn::Tensor({{0.0f}}, {1, 1}),
        nn::Tensor({{1.0f}}, {1, 1}),
        nn::Tensor({{1.0f}}, {1, 1}),
        nn::Tensor({{0.0f}}, {1, 1})
    };
    
    // Create loss function
    nn::MSELoss loss_fn;
    
    // Train the network for multiple epochs
    std::cout << "Starting training..." << std::endl;
    const int epochs = 10000;
    const float learning_rate = 0.5f;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < train_data.size(); ++i) {
            nn::Tensor output = net.forward(train_data[i]);
            float loss = loss_fn.compute(output, train_targets[i]);
            total_loss += loss;
            
            // Simple gradient descent step
            net.train_step(train_data[i], train_targets[i], learning_rate);
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Average Loss: " << (total_loss / train_data.size()) << std::endl;
        }
    }
    
    // Test the trained network
    std::cout << "\nTesting the trained network:" << std::endl;
    for (size_t i = 0; i < train_data.size(); ++i) {
        nn::Tensor output = net.forward(train_data[i]);
        std::cout << "Input: [" << train_data[i](0, 0) << ", " << train_data[i](1, 0) 
                  << "] -> Output: " << output(0, 0) << ", Target: " << train_targets[i](0, 0) << std::endl;
    }
    
    // Evaluate final performance
    float total_error = 0.0f;
    for (size_t i = 0; i < train_data.size(); ++i) {
        nn::Tensor output = net.forward(train_data[i]);
        float diff = output(0, 0) - train_targets[i](0, 0);
        total_error += diff * diff;
    }
    float mse = total_error / train_data.size();
    std::cout << "\nFinal MSE: " << mse << std::endl;
    
    if (mse < 0.01f) {
        std::cout << "Training successful! Network learned the XOR function." << std::endl;
    } else {
        std::cout << "Training could be improved. Final MSE: " << mse << std::endl;
    }
    
    return 0;
}