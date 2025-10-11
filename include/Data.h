#ifndef DATA_H
#define DATA_H

#include "Tensor.h"
#include <vector>
#include <string>

namespace nn {

class DataLoader {
public:
    DataLoader(const std::vector<Tensor>& data, const std::vector<Tensor>& labels, size_t batch_size = 1);
    
    struct Batch {
        std::vector<Tensor> data;
        std::vector<Tensor> labels;
    };
    
    bool has_next() const;
    Batch get_next();
    void reset();
    
private:
    std::vector<Tensor> data_;
    std::vector<Tensor> labels_;
    size_t batch_size_;
    size_t current_index_;
};

// Utility function to load data from a simple format (for testing)
std::pair<std::vector<Tensor>, std::vector<Tensor>> load_simple_dataset();

// Utility functions for data preprocessing
Tensor normalize(const Tensor& tensor);
Tensor standardize(const Tensor& tensor);
std::vector<Tensor> normalize_dataset(const std::vector<Tensor>& dataset);
std::vector<Tensor> standardize_dataset(const std::vector<Tensor>& dataset);

} // namespace nn

#endif // DATA_H