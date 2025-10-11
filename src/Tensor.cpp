#include "Tensor.h"

namespace nn {

// Constructors
Tensor::Tensor() : data_(), shape_() {}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape)
    : data_(data), shape_(shape) {
    if (data.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) {
        throw std::runtime_error("Data size does not match shape");
    }
}

Tensor::Tensor(const std::vector<std::vector<float>>& data) {
    if (!data.empty() && !data[0].empty()) {
        shape_ = {data.size(), data[0].size()};
        data_.reserve(shape_[0] * shape_[1]);
        for (const auto& row : data) {
            if (row.size() != shape_[1]) {
                throw std::runtime_error("All rows must have the same size");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }
}

Tensor::Tensor(size_t rows, size_t cols)
    : shape_({rows, cols}), data_(rows * cols, 0.0f) {}

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : data_(other.data_), shape_(other.shape_) {}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
    }
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)) {}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
    }
    return *this;
}

// Access operators
float& Tensor::operator()(size_t row, size_t col) {
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index(row, col)];
}

const float& Tensor::operator()(size_t row, size_t col) const {
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index(row, col)];
}

float& Tensor::operator[](size_t index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

// Helper function to calculate linear index
size_t Tensor::index(size_t row, size_t col) const {
    return row * shape_[1] + col;
}

// Arithmetic operations
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes do not match for addition");
    }
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes do not match for subtraction");
    }
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes do not match for element-wise multiplication");
    }
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::runtime_error("Division by zero");
    }
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

// Matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
    if (cols() != other.rows()) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }
    Tensor result(rows(), other.cols());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols(); ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Transpose
Tensor Tensor::transpose() const {
    Tensor result(cols(), rows());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Activation functions
Tensor Tensor::sigmoid() const {
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}

Tensor Tensor::tanh_activation() const {
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    return result;
}

// Utility functions
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

Tensor Tensor::sum(int axis) const {
    if (axis == -1) {
        float total = std::accumulate(data_.begin(), data_.end(), 0.0f);
        return Tensor({total}, {1, 1});
    } else if (axis == 0) {
        Tensor result(1, cols());
        for (size_t j = 0; j < cols(); ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < rows(); ++i) {
                sum += (*this)(i, j);
            }
            result(0, j) = sum;
        }
        
        return result;
    } else if (axis == 1) {
        Tensor result(rows(), 1);
        for (size_t i = 0; i < rows(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols(); ++j) {
                sum += (*this)(i, j);
            }
            result(i, 0) = sum;
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis for sum operation");
}

void Tensor::print() const {
    std::cout << "[";
    for (size_t i = 0; i < rows(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols(); ++j) {
            std::cout << (*this)(i, j);
            if (j < cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < rows() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (total_size != data_.size()) {
        throw std::runtime_error("Reshape size does not match total number of elements");
    }
    shape_ = new_shape;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

} // namespace nn
 
