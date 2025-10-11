#!/bin/bash

# Build script for Neural Network Library

echo "Building Neural Network Library..."

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Change to build directory
cd build

# Configure with CMake
cmake ..

# Build the project
make

echo "Build completed! Executables are in the build/ directory."
echo "Available executables:"
echo "  - ./main: Basic example"
echo "  - ./xor_example: XOR problem example"
echo "  - ./mnist_example: MNIST-like problem example"
echo "  - ./tests: Run unit tests"