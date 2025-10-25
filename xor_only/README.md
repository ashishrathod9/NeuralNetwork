# XOR Neural Network Example

This is a simplified version of the neural network library focused only on solving the XOR problem.

## Overview
This example demonstrates a neural network learning the XOR function using:
- A 2-4-4-1 network architecture (2 inputs, 2 hidden layers of 4 neurons each, 1 output)
- Linear layers with Sigmoid activation functions
- Mean Squared Error (MSE) loss function
- Simple gradient descent for training

## Building and Running

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Configure with CMake:
   ```bash
   cmake ..
   ```

3. Build the project:
   ```bash
   make
   ```

4. Run the XOR example:
   ```bash
   ./xor_example
   ```

## Expected Output
The network should learn to approximate the XOR function:
- Input [0, 0] -> Output near 0
- Input [0, 1] -> Output near 1
- Input [1, 0] -> Output near 1
- Input [1, 1] -> Output near 0

The training will show the decreasing loss over epochs, and at the end, it will show the final test results.