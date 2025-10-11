#include "Training.h"
#include <random>
#include <algorithm>
#include <iostream>

namespace nn {

Trainer::Trainer(TrainingConfig config)
    : config_(config)
{}

void Trainer::train(Network& network,
                    const std::vector<Tensor>& train_data,
                    const std::vector<Tensor>& train_targets,
                    Loss& loss_fn,
                    Optimizer& optimizer) {

    if (train_data.size() != train_targets.size()) {
        throw std::runtime_error("Training data and targets must have the same size");
    }

    // Calculate validation split
    size_t val_size = static_cast<size_t>(train_data.size() * config_.validation_split);
    size_t train_size = train_data.size() - val_size;

    // Create indices and shuffle for random splitting
    std::vector<size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Split data
    std::vector<Tensor> train_subset, val_subset;
    std::vector<Tensor> train_targets_subset, val_targets_subset;

    for (size_t i = 0; i < train_size; ++i) {
        train_subset.push_back(train_data[indices[i]]);
        train_targets_subset.push_back(train_targets[indices[i]]);
    }
    for (size_t i = train_size; i < train_data.size(); ++i) {
        val_subset.push_back(train_data[indices[i]]);
        val_targets_subset.push_back(train_targets[indices[i]]);
    }

    // Training loop
    for (size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        float total_loss = 0.0f;

        // Process in batches
        size_t num_batches = (train_subset.size() + config_.batch_size - 1) / config_.batch_size;

        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            size_t start_idx = batch_idx * config_.batch_size;
            size_t end_idx = std::min(start_idx + config_.batch_size, train_subset.size());

            float batch_loss = 0.0f;

            for (size_t i = start_idx; i < end_idx; ++i) {
                float loss = network.train(train_subset[i], train_targets_subset[i], loss_fn, optimizer);
                batch_loss += loss;
            }

            // Average the loss for the batch
            batch_loss /= (end_idx - start_idx);
            total_loss += batch_loss;
        }

        float avg_loss = total_loss / num_batches;

        if (config_.verbose && epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << "/" << config_.epochs
                      << " - Loss: " << avg_loss;

            // Validate every 10 epochs
            if (!val_subset.empty()) {
                float val_loss = validate(network, val_subset, val_targets_subset, loss_fn);
                std::cout << " - Val Loss: " << val_loss;
            }
            std::cout << std::endl;
        }
    }
}

float Trainer::validate(Network& network,
                        const std::vector<Tensor>& val_data,
                        const std::vector<Tensor>& val_targets,
                        Loss& loss_fn) {

    if (val_data.size() != val_targets.size()) {
        throw std::runtime_error("Validation data and targets must have the same size");
    }

    float total_loss = 0.0f;
    for (size_t i = 0; i < val_data.size(); ++i) {
        total_loss += network.evaluate(val_data[i], val_targets[i], loss_fn);
    }

    return total_loss / val_data.size();
}

} // namespace nn
