#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace nn {

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape);
    Tensor(const std::vector<std::vector<float>>& data);  // For 2D convenience
    Tensor(size_t rows, size_t cols);  // Initialize with zeros

    // Copy constructor and assignment operator
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // Move constructor and assignment operator (for simplicity we'll just copy)
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor
    ~Tensor() = default;

    // Accessors
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    size_t rows() const { return shape_[0]; }
    size_t cols() const { return shape_.size() > 1 ? shape_[1] : 1; }
    size_t size() const { return data_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }

    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(const Tensor& other) const;  // Element-wise multiplication
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;

    // Activation functions
    Tensor sigmoid() const;
    Tensor relu() const;

    // Utility functions
    void fill(float value);
    Tensor sum(int axis = -1) const;  // Sum along axis (-1 for all elements)

    // Print tensor
    void print() const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;

    void reshape(const std::vector<size_t>& new_shape);
    size_t index(size_t row, size_t col) const;
};

// Non-member operators
Tensor operator*(float scalar, const Tensor& tensor);

} // namespace nn

#endif // TENSOR_H