#include "Network.h"
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Training.h"
#include <iostream>
#include <vector>
#include <random>

// Function to generate synthetic MNIST-like data for demonstration
std::pair<std::vector<nn::Tensor>, std::vector<nn::Tensor>> generate_mnist_data(size_t num_samples = 100) {
    std::vector<nn::Tensor> data;
    std::vector<nn::Tensor> labels;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pixel_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);
    
    for (size_t i = 0; i < num_samples; ++i) {
        // Create a 28x28 "image" (flattened to 784 elements)
        std::vector<float> image_data(784);
        for (size_t j = 0; j < 784; ++j) {
            image_data[j] = pixel_dist(gen);
        }
        
        // Create one-hot encoded label
        std::vector<float> label_data(10, 0.0f);
        int label = label_dist(gen);
        label_data[label] = 1.0f;
        
        data.push_back(nn::Tensor(image_data, {784, 1}));
        labels.push_back(nn::Tensor(label_data, {10, 1}));
    }
    
    return std::make_pair(data, labels);
}

int main() {
    std::cout << "Neural Network Library - MNIST Example (Synthetic Data)" << std::endl;
    std::cout << "Training a network on synthetic MNIST-like data" << std::endl;
    
    // Generate synthetic MNIST-like training data
    std::cout << "Generating synthetic training data..." << std::endl;
    auto [train_data, train_labels] = generate_mnist_data(1000);  // 1000 samples
    auto [test_data, test_labels] = generate_mnist_data(200);     // 200 test samples
    
    std::cout << "Training samples: " << train_data.size() << std::endl;
    std::cout << "Test samples: " << test_data.size() << std::endl;
    
    // Create a network: 784 -> 128 -> 64 -> 10
    nn::Network net;
    
    // Add layers: Input(784) -> Hidden(128) -> Hidden(64) -> Output(10)
    net.add_layer(new nn::Linear(784, 128));
    net.add_layer(new nn::ReLU());      // Activation after first linear layer
    net.add_layer(new nn::Linear(128, 64));
    net.add_layer(new nn::ReLU());      // Activation after second linear layer
    net.add_layer(new nn::Linear(64, 10));
    // Note: No activation after final layer for raw logits
    
    // Create loss function and optimizer
    nn::CrossEntropyLoss loss_fn;
    nn::Adam optimizer(0.001f);  // Using Adam optimizer with learning rate 0.001
    
    // Create trainer with configuration
    nn::TrainingConfig config(10, 32, 0.001f, true, 0.2f);  // 10 epochs, batch size 32, 20% validation
    nn::Trainer trainer(config);
    
    // Train the network
    std::cout << "\nStarting training..." << std::endl;
    trainer.train(net, train_data, train_labels, loss_fn, optimizer);
    
    // Evaluate the network on test data
    std::cout << "\nEvaluating on test data..." << std::endl;
    float total_loss = 0.0f;
    int correct_predictions = 0;
    int total_predictions = 0;
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        nn::Tensor output = net.forward(test_data[i]);
        total_loss += loss_fn.compute_loss(output, test_labels[i]);
        
        // Find predicted class (index of max value)
        float max_val = output(0, 0);
        int predicted_class = 0;
        for (int j = 1; j < 10; ++j) {
            if (output(j, 0) > max_val) {
                max_val = output(j, 0);
                predicted_class = j;
            }
        }
        
        // Find actual class
        int actual_class = 0;
        for (int j = 0; j < 10; ++j) {
            if (test_labels[i](j, 0) == 1.0f) {
                actual_class = j;
                break;
            }
        }
        
        if (predicted_class == actual_class) {
            correct_predictions++;
        }
        total_predictions++;
    }
    
    float avg_loss = total_loss / test_data.size();
    float accuracy = static_cast<float>(correct_predictions) / total_predictions * 100.0f;
    
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Average Loss: " << avg_loss << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Correct predictions: " << correct_predictions << "/" << total_predictions << std::endl;
    
    // Clean up allocated memory
    auto& layers = net.get_layers();
    for (auto* layer : layers) {
        delete layer;
    }
    
    return 0;
}