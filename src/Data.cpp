#include "Data.h"

namespace nn {

DataLoader::DataLoader(const std::vector<Tensor>& data, const std::vector<Tensor>& labels, size_t batch_size)
    : data_(data), labels_(labels), batch_size_(batch_size), current_index_(0) {
    if (data.size() != labels.size()) {
        throw std::runtime_error("Data and label sets must have the same size");
    }
}

bool DataLoader::has_next() const {
    return current_index_ < data_.size();
}

DataLoader::Batch DataLoader::get_next() {
    if (!has_next()) {
        throw std::runtime_error("No more data to load");
    }
    
    Batch batch;
    size_t end_index = std::min(current_index_ + batch_size_, data_.size());
    
    for (size_t i = current_index_; i < end_index; ++i) {
        batch.data.push_back(data_[i]);
        batch.labels.push_back(labels_[i]);
    }
    
    current_index_ = end_index;
    return batch;
}

void DataLoader::reset() {
    current_index_ = 0;
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> load_simple_dataset() {
    // Example: XOR problem dataset
    std::vector<Tensor> data = {
        Tensor({{0.0f, 0.0f}}, {2, 1}),
        Tensor({{0.0f, 1.0f}}, {2, 1}),
        Tensor({{1.0f, 0.0f}}, {2, 1}),
        Tensor({{1.0f, 1.0f}}, {2, 1})
    };
    
    std::vector<Tensor> labels = {
        Tensor({{0.0f}}, {1, 1}),
        Tensor({{1.0f}}, {1, 1}),
        Tensor({{1.0f}}, {1, 1}),
        Tensor({{0.0f}}, {1, 1})
    };
    
    return std::make_pair(data, labels);
}

Tensor normalize(const Tensor& tensor) {
    // Find min and max values
    float min_val = tensor[0];
    float max_val = tensor[0];
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] < min_val) min_val = tensor[i];
        if (tensor[i] > max_val) max_val = tensor[i];
    }
    
    // Normalize to [0, 1]
    float range = max_val - min_val;
    if (range == 0.0f) range = 1.0f;  // Avoid division by zero
    
    Tensor result(tensor.rows(), tensor.cols());
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = (tensor[i] - min_val) / range;
    }
    
    return result;
}

Tensor standardize(const Tensor& tensor) {
    // Calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.size(); ++i) {
        sum += tensor[i];
    }
    float mean = sum / tensor.size();
    
    // Calculate std deviation
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < tensor.size(); ++i) {
        float diff = tensor[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float std_dev = std::sqrt(sum_sq_diff / tensor.size());
    if (std_dev == 0.0f) std_dev = 1.0f;  // Avoid division by zero
    
    // Standardize
    Tensor result(tensor.rows(), tensor.cols());
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = (tensor[i] - mean) / std_dev;
    }
    
    return result;
}

std::vector<Tensor> normalize_dataset(const std::vector<Tensor>& dataset) {
    std::vector<Tensor> normalized;
    for (const auto& tensor : dataset) {
        normalized.push_back(normalize(tensor));
    }
    return normalized;
}

std::vector<Tensor> standardize_dataset(const std::vector<Tensor>& dataset) {
    std::vector<Tensor> standardized;
    for (const auto& tensor : dataset) {
        standardized.push_back(standardize(tensor));
    }
    return standardized;
}

} // namespace nn