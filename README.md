# Neural Network Library

A lightweight, modular deep learning library implemented from scratch in C++. This library allows users to define, train, and evaluate neural networks with common layers and optimizers, with a focus on extensibility and educational value.

## Features

- **Core Architecture**: Feed-forward networks with configurable layers
- **Layers & Activations**: Linear (fully connected) layers with Sigmoid, ReLU, and Tanh activations
- **Loss Functions**: MSE (Mean Squared Error) and Cross-Entropy loss functions
- **Backpropagation**: Manual implementation of forward and backward propagation
- **Optimizers**: SGD, Momentum, and Adam optimizers
- **Training Loop**: Batch processing with epoch monitoring
- **Data Handling**: Utilities for data loading, normalization, and preprocessing
- **Performance**: Clean memory management and efficient matrix operations

## Requirements

- C++17 or later
- CMake 3.10 or later
- (Optional) OpenMP for parallel computation

## Building

1. Clone the repository
2. Create a build directory:

```bash
mkdir build
cd build
```

3. Configure with CMake:

```bash
cmake ..
```

4. Build the project:

```bash
make
```

## Usage Examples

### Basic Example

```cpp
#include "Network.h"
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"

// Create a network: 2 -> 4 -> 1
nn::Network net;
net.add_layer(new nn::Linear(2, 4));  // Input: 2 features, Hidden: 4 neurons
net.add_layer(new nn::Sigmoid());    // Activation function
net.add_layer(new nn::Linear(4, 1)); // Hidden: 4 neurons, Output: 1 neuron
net.add_layer(new nn::Sigmoid());    // Output activation

// Define loss function and optimizer
nn::MSELoss loss_fn;
nn::SGD optimizer(0.1f);  // Learning rate = 0.1

// Create sample data
nn::Tensor input({{0.5f, 0.3f}}, {2, 1});
nn::Tensor target({{0.8f}}, {1, 1});

// Train the network
float loss = net.train(input, target, loss_fn, optimizer);

// Clean up
auto& layers = net.get_layers();
for (auto* layer : layers) {
    delete layer;
}
```

### Using Training Utilities

```cpp
#include "Training.h"

// Prepare training data
std::vector<nn::Tensor> train_data = {/* your data */};
std::vector<nn::Tensor> train_targets = {/* your targets */};

// Configure training
nn::TrainingConfig config(100, 32, 0.01f, true, 0.2f);  // 100 epochs, batch size 32, etc.
nn::Trainer trainer(config);

// Train the network
trainer.train(net, train_data, train_targets, loss_fn, optimizer);
```

## Architecture

The library is organized into several key components:

- `Tensor`: Core data structure for matrix operations
- `Layer`: Base class for network layers (Linear, Activation functions)
- `Loss`: Loss function implementations (MSE, Cross-Entropy)
- `Optimizer`: Parameter update methods (SGD, Momentum, Adam)
- `Network`: Container for layers with forward/backward propagation
- `Training`: High-level training utilities

## Layers

### Linear Layer

Fully connected layer that applies: `output = weights * input + bias`

```cpp
nn::Linear layer(input_size, output_size);
```

### Activation Functions

- `Sigmoid`: Sigmoid activation function
- `ReLU`: Rectified Linear Unit activation function  
- `Tanh`: Hyperbolic tangent activation function

## Optimizers

### SGD (Stochastic Gradient Descent)

Basic optimizer with configurable learning rate:

```cpp
nn::SGD optimizer(learning_rate);
```

### Momentum

SGD with momentum:

```cpp
nn::Momentum optimizer(learning_rate, momentum);
```

### Adam

Adaptive Moment Estimation optimizer:

```cpp
nn::Adam optimizer(learning_rate, beta1, beta2, epsilon);
```

## Examples

The repository includes several example programs:

- `main.cpp`: Basic XOR problem example
- `xor_example.cpp`: More comprehensive XOR problem with hidden layers
- `mnist_example.cpp`: MNIST-like problem with synthetic data

## Extending the Library

The library is designed to be easily extensible:

1. **New Layers**: Inherit from the `Layer` base class and implement the required methods
2. **New Loss Functions**: Inherit from the `Loss` base class
3. **New Optimizers**: Inherit from the `Optimizer` base class
4. **New Activation Functions**: Inherit from the `Layer` base class (like existing activations)

## Testing

Run the unit tests with:

```bash
./tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.