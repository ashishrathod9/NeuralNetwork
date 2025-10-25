#include "Tensor.h"
#include <random>
#include <iostream>

namespace nn {

Tensor::Tensor() : data_(), shape_({0, 0}) {}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape) 
    : data_(data), shape_(shape) {
    if (data.size() != shape[0] * shape[1]) {
        throw std::runtime_error("Data size does not match shape");
    }
}

Tensor::Tensor(const std::vector<std::vector<float>>& data) {
    if (data.empty()) {
        shape_ = {0, 0};
        return;
    }
    shape_ = {data.size(), data[0].size()};
    data_.reserve(shape_[0] * shape_[1]);
    
    for (const auto& row : data) {
        for (float val : row) {
            data_.push_back(val);
        }
    }
}

Tensor::Tensor(size_t rows, size_t cols) 
    : shape_({rows, cols}) {
    data_.resize(rows * cols, 0.0f);
}

Tensor::Tensor(const Tensor& other) 
    : data_(other.data_), shape_(other.shape_) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
    }
    return *this;
}

float& Tensor::operator()(size_t row, size_t col) {
    return data_[index(row, col)];
}

const float& Tensor::operator()(size_t row, size_t col) const {
    return data_[index(row, col)];
}

float& Tensor::operator[](size_t index) {
    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
    return data_[index];
}

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

Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result(shape_[0], shape_[1]);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - scalar;
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

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_[1] != other.shape_[0]) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    size_t rows = shape_[0];
    size_t cols = other.shape_[1];
    size_t inner = shape_[1];
    
    Tensor result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < inner; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

Tensor Tensor::transpose() const {
    Tensor result(shape_[1], shape_[0]);
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

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

void Tensor::fill(float value) {
    for (auto& val : data_) {
        val = value;
    }
}

Tensor Tensor::sum(int axis) const {
    if (axis == -1) {
        // Sum all elements
        float total = 0.0f;
        for (const auto& val : data_) {
            total += val;
        }
        return Tensor({{total}}, {1, 1});
    } else if (axis == 0) {
        // Sum along rows (reduce rows)
        Tensor result(1, shape_[1]);
        for (size_t j = 0; j < shape_[1]; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < shape_[0]; ++i) {
                sum += (*this)(i, j);
            }
            result(0, j) = sum;
        }
        return result;
    } else { // axis == 1
        // Sum along columns (reduce columns)
        Tensor result(shape_[0], 1);
        for (size_t i = 0; i < shape_[0]; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < shape_[1]; ++j) {
                sum += (*this)(i, j);
            }
            result(i, 0) = sum;
        }
        return result;
    }
}

void Tensor::print() const {
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    if (new_shape[0] * new_shape[1] != data_.size()) {
        throw std::runtime_error("New shape incompatible with data size");
    }
    shape_ = new_shape;
}

size_t Tensor::index(size_t row, size_t col) const {
    return row * shape_[1] + col;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

} // namespace nn