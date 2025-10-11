#include "Tensor.h"
#include "Network.h"
#include "Training.h"
#include "Loss.h"
#include "Optimizer.h"
#include <iostream>
#include <vector>
#include <string>

class BackendTrainer {
public:
    BackendTrainer() : lossFunction(), optimizer(0.1f) {
        setupNetwork(4); // Default to 4 hidden neurons
    }

    void setupNetwork(int hiddenSize) {
        // Clear existing layers
        auto& existingLayers = network.get_layers();
        for (auto* layer : existingLayers) {
            delete layer;
        }
        
        // Create a simple network: 2 -> hidden -> 1 (like in the original main.cpp example)
        nn::Linear* inputLayer = new nn::Linear(2, hiddenSize);    // 2 inputs -> hiddenSize outputs
        nn::Linear* outputLayer = new nn::Linear(hiddenSize, 1);   // hiddenSize inputs -> 1 output
        
        network.add_layer(inputLayer);
        network.add_layer(outputLayer);
    }

    void setupTrainingData(const std::string& dataset) {
        trainingData.clear();

        // Using vectors like in the original example, then converting to tensors
        std::vector<std::vector<float>> inputs_data;
        std::vector<std::vector<float>> targets_data;

        if (dataset == "XOR" || dataset == "xor") {
            // XOR training data
            inputs_data = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
            targets_data = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
        }
        else if (dataset == "AND" || dataset == "and") {
            // AND training data
            inputs_data = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
            targets_data = {{0.0f}, {0.0f}, {0.0f}, {1.0f}};
        }
        else if (dataset == "OR" || dataset == "or") {
            // OR training data
            inputs_data = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
            targets_data = {{0.0f}, {1.0f}, {1.0f}, {1.0f}};
        }
        else { // Default to XOR
            setupTrainingData("XOR");
            return;
        }

        // Convert to tensors like in the original main.cpp
        std::vector<nn::Tensor> inputs, targets;
        for (const auto& input : inputs_data) {
            inputs.push_back(nn::Tensor(input, {input.size(), 1}));
        }
        for (const auto& target : targets_data) {
            targets.push_back(nn::Tensor(target, {target.size(), 1}));
        }
        
        // Store as pairs
        for (size_t i = 0; i < inputs.size(); ++i) {
            trainingData.push_back(std::make_pair(inputs[i], targets[i]));
        }
    }

    void train(int epochs, float learningRate, int hiddenSize, const std::string& dataset) {
        setupNetwork(hiddenSize);
        setupTrainingData(dataset);
        optimizer = nn::SGD(learningRate);

        std::cout << "Starting training for " << epochs << " epochs with " << hiddenSize 
                  << " hidden neurons on " << dataset << " dataset..." << std::endl;

        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0.0f;
            for (auto& dataPair : trainingData) {
                float loss = network.train(dataPair.first, dataPair.second, lossFunction, optimizer);
                totalLoss += loss;
            }
            
            float avgLoss = totalLoss / trainingData.size();
            
            if (epoch % 20 == 0 || epoch == epochs - 1) { // Log every 20 epochs or last epoch
                std::cout << "Epoch " << epoch << ", Average Loss: " << avgLoss << std::endl;
            }
        }

        std::cout << "Training completed after " << epochs << " epochs" << std::endl;
    }

private:
    nn::Network network;
    nn::MSELoss lossFunction;
    nn::SGD optimizer;
    std::vector<std::pair<nn::Tensor, nn::Tensor>> trainingData;
};

int main() {
    std::cout << "Neural Network Backend Trainer" << std::endl;
    
    BackendTrainer trainer;
    
    // Example training run
    int epochs = 100;
    float learningRate = 0.1f;
    int hiddenSize = 4;
    std::string dataset = "XOR";
    
    trainer.train(epochs, learningRate, hiddenSize, dataset);
    
    return 0;
}