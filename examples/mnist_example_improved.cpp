#include "Network.h"
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Function to generate synthetic MNIST-like data
class SyntheticMNISTGenerator {
private:
    std::mt19937 rng;
    
public:
    SyntheticMNISTGenerator(unsigned seed = 42) : rng(seed) {}
    
    // Generate a synthetic digit pattern (simplified)
    std::vector<float> generateDigit(int digit) {
        std::vector<float> image(784, 0.0f); // 28x28 = 784 pixels
        
        // Simple patterns for digits 0-9
        switch(digit) {
            case 0: // Circle for 0
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 5 && i < 23 && j > 5 && j < 23) && 
                            ((i < 8 || i > 20) || (j < 8 || j > 20))) {
                            image[i * 28 + j] = 0.8f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.2f;
                        }
                    }
                }
                break;
                
            case 1: // Vertical line for 1
                for (int i = 5; i < 25; i++) {
                    int j = 14; // Middle column
                    image[i * 28 + j] = 0.8f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.2f;
                    if (i > 6 && i < 10) image[i * 28 + j-2] = 0.6f;
                    if (i == 24) {
                        for (int k = 12; k < 17; k++) image[i * 28 + k] = 0.7f;
                    }
                }
                break;
                
            case 2: // Simple 2
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||  // Top curve
                            (i > 9 && i < 19 && ((i+j) > 20 && (i+j) < 35)) ||  // Diagonal
                            (i > 19 && i < 24 && j > 4 && j < 24)) {  // Bottom curve
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 3: // Simple 3
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||   // Top curve
                            (i > 11 && i < 17 && j > 15 && j < 24) || // Middle right
                            (i > 19 && i < 24 && j > 4 && j < 24)) {  // Bottom curve
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 4: // Simple 4
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((j > 8 && j < 12 && i > 4 && i < 20) ||   // Vertical line
                            (i > 14 && i < 18 && j > 4 && j < 20) ||  // Middle horizontal
                            (j > 18 && j < 22 && i > 4 && i < 18)) {  // Right vertical
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 5: // Simple 5
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||     // Top horizontal
                            (i > 8 && i < 19 && j > 4 && j < 8) ||     // Left vertical
                            (i > 11 && i < 16 && j > 4 && j < 24) ||   // Middle horizontal
                            (i > 15 && i < 24 && j > 19 && j < 24) ||  // Right bottom vertical
                            (i > 19 && i < 24 && j > 4 && j < 20)) {   // Bottom horizontal
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 6: // Simple 6
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||     // Top
                            (i > 4 && i < 24 && j > 4 && j < 8) ||     // Left vertical
                            (i > 19 && i < 24 && j > 4 && j < 24) ||   // Bottom
                            (i > 19 && i < 24 && j > 19 && j < 24) ||  // Bottom right
                            (i > 9 && i < 20 && j > 19 && j < 24)) {   // Right vertical
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 7: // Simple 7
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||    // Top horizontal
                            (i > 8 && i < 24 && j > 4 && j < 9) ||    // Bottom left to right
                            (i+j > 30 && i < 24 && j > 15)) {         // Diagonal
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 8: // Simple 8
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||     // Top
                            (i > 4 && i < 24 && j > 4 && j < 8) ||     // Left upper
                            (i > 19 && i < 24 && j > 4 && j < 24) ||   // Bottom
                            (i > 4 && i < 24 && j > 19 && j < 24) ||   // Right
                            (i > 11 && i < 17 && j > 4 && j < 24)) {   // Middle
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
                
            case 9: // Simple 9
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        if ((i > 4 && i < 9 && j > 4 && j < 24) ||     // Top
                            (i > 4 && i < 19 && j > 4 && j < 8) ||     // Left
                            (i > 4 && i < 19 && j > 19 && j < 24) ||   // Right upper
                            (i > 11 && i < 19 && j > 4 && j < 24) ||   // Middle
                            (i > 19 && i < 24 && j > 4 && j < 24)) {   // Bottom
                            image[i * 28 + j] = 0.7f + (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.3f;
                        }
                    }
                }
                break;
        }
        
        // Add some noise to make it more realistic
        for (float& pixel : image) {
            if (pixel > 0) {
                pixel += (static_cast<float>(rng()) / static_cast<float>(rng.max())) * 0.1f - 0.05f;
                pixel = std::max(0.0f, std::min(1.0f, pixel)); // Clamp between 0 and 1
            }
        }
        
        return image;
    }
    
    // Generate training data
    std::pair<std::vector<nn::Tensor>, std::vector<nn::Tensor>> generateDataset(int numSamples) {
        std::vector<nn::Tensor> inputs, targets;
        
        for (int i = 0; i < numSamples; i++) {
            int digit = i % 10; // Cycle through digits 0-9
            std::vector<float> image = generateDigit(digit);
            
            // Create input tensor (28x28 flattened to 784x1)
            inputs.push_back(nn::Tensor(image, {784, 1}));
            
            // Create one-hot encoded target
            std::vector<float> target(10, 0.0f);
            target[digit] = 1.0f;
            targets.push_back(nn::Tensor(target, {10, 1}));
        }
        
        return std::make_pair(inputs, targets);
    }
};

int main() {
    std::cout << "Neural Network Library - Improved MNIST Example" << std::endl;
    std::cout << "Training a network on synthetic MNIST-like data" << std::endl;
    
    // Generate synthetic training data
    std::cout << "Generating synthetic training data..." << std::endl;
    SyntheticMNISTGenerator generator(42);
    
    // Generate training data (60,000 samples would be typical, but we'll use 6000 for efficiency)
    auto [train_inputs, train_targets] = generator.generateDataset(6000);
    auto [test_inputs, test_targets] = generator.generateDataset(1000);
    
    std::cout << "Training samples: " << train_inputs.size() << std::endl;
    std::cout << "Test samples: " << test_inputs.size() << std::endl;
    
    // Create a proper network for MNIST (784 -> 128 -> 64 -> 10)
    nn::Network net;
    
    // Add layers - using ReLU activation for better MNIST performance
    net.add_layer(new nn::Linear(784, 128));  // Input: 784 pixels, Hidden: 128 neurons
    net.add_layer(new nn::ReLU());             // Activation function
    net.add_layer(new nn::Linear(128, 64));   // Hidden: 128 -> 64
    net.add_layer(new nn::ReLU());             // Activation function
    net.add_layer(new nn::Linear(64, 10));    // Output: 64 -> 10 classes
    
    // Create loss function and optimizer
    nn::CrossEntropyLoss loss_fn;  // Better for classification
    nn::SGD optimizer(0.01f);      // Lower learning rate for stability
    
    std::cout << "\nStarting training..." << std::endl;
    
    // Training loop with more epochs and better parameters
    const int epochs = 500;
    const int batchSize = 32;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        // Shuffle training data indices
        std::vector<int> indices(train_inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 local_rng(rd());
        std::shuffle(indices.begin(), indices.end(), local_rng);
        
        // Train on shuffled data
        for (size_t i = 0; i < train_inputs.size(); i++) {
            int idx = indices[i];
            float loss = net.train(train_inputs[idx], train_targets[idx], loss_fn, optimizer);
            total_loss += loss;
        }
        
        // Print progress every 50 epochs
        if (epoch % 50 == 0) {
            float avg_loss = total_loss / train_inputs.size();
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << " - Average Loss: " << avg_loss << std::endl;
        }
    }
    
    std::cout << "\nEvaluating on test data..." << std::endl;
    
    // Evaluate the trained network
    int correct = 0;
    int total = test_inputs.size();
    float total_test_loss = 0.0f;
    
    for (int i = 0; i < test_inputs.size(); i++) {
        nn::Tensor output = net.forward(test_inputs[i]);
        
        // Find predicted class (highest output value)
        int predicted = 0;
        float max_val = output(0, 0);
        for (int j = 1; j < 10; j++) {
            if (output(j, 0) > max_val) {
                max_val = output(j, 0);
                predicted = j;
            }
        }
        
        // Find actual class (from target)
        int actual = 0;
        max_val = test_targets[i](0, 0);
        for (int j = 1; j < 10; j++) {
            if (test_targets[i](j, 0) > max_val) {
                max_val = test_targets[i](j, 0);
                actual = j;
            }
        }
        
        // Check if prediction is correct
        if (predicted == actual) {
            correct++;
        }
        
        // Calculate loss
        total_test_loss += loss_fn.compute_loss(output, test_targets[i]);
    }
    
    float accuracy = (float)correct / total * 100.0f;
    
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Average Loss: " << total_test_loss / total << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Correct predictions: " << correct << "/" << total << std::endl;
    
    std::cout << "\nSample predictions:" << std::endl;
    for (int i = 0; i < 10; i++) {
        nn::Tensor output = net.forward(test_inputs[i]);
        
        // Find predicted class
        int predicted = 0;
        float max_val = output(0, 0);
        for (int j = 1; j < 10; j++) {
            if (output(j, 0) > max_val) {
                max_val = output(j, 0);
                predicted = j;
            }
        }
        
        // Find actual class
        int actual = 0;
        max_val = test_targets[i](0, 0);
        for (int j = 1; j < 10; j++) {
            if (test_targets[i](j, 0) > max_val) {
                max_val = test_targets[i](j, 0);
                actual = j;
            }
        }
        
        std::cout << "Input digit: " << actual 
                  << ", Predicted: " << predicted 
                  << ", " << (predicted == actual ? "✓" : "✗") << std::endl;
    }
    
    // Clean up allocated memory
    auto& layers = net.get_layers();
    for (auto* layer : layers) {
        delete layer;
    }
    
    std::cout << "\nTraining and evaluation completed!" << std::endl;
    
    return 0;
}