#include "Tensor.h"
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Network.h"
#include "Training.h"
#include <iostream>
#include <cassert>

void test_tensor_operations() {
    std::cout << "Testing Tensor operations..." << std::endl;
    
    // Create a 2x3 tensor
    nn::Tensor a(std::vector<std::vector<float>>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    std::cout << "Tensor A:" << std::endl;
    a.print();
    
    // Test element access
    assert(a(0, 1) == 2.0f);
    std::cout << "Element (0,1): " << a(0, 1) << std::endl;
    
    // Test matrix multiplication
    nn::Tensor b({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    std::cout << "Tensor B:" << std::endl;
    b.print();
    
    nn::Tensor c = a.matmul(b);
    std::cout << "A * B:" << std::endl;
    c.print();
    
    // Test scalar multiplication
    nn::Tensor d = a * 2.0f;
    std::cout << "A * 2:" << std::endl;
    d.print();
    
    // Test transpose
    nn::Tensor e = a.transpose();
    std::cout << "A transpose:" << std::endl;
    e.print();
    
    std::cout << "Tensor operations test completed." << std::endl << std::endl;
}

void test_linear_layer() {
    std::cout << "Testing Linear layer..." << std::endl;
    
    // Create a linear layer: 3 inputs -> 2 outputs
    nn::Linear layer(3, 2);
    
    // Create input tensor (3x1)
    nn::Tensor input({{1.0f, 2.0f, 3.0f}}, {3, 1});
    std::cout << "Input:" << std::endl;
    input.print();
    
    // Forward pass
    nn::Tensor output = layer.forward(input);
    std::cout << "Output from linear layer:" << std::endl;
    output.print();
    
    // Test backward pass
    nn::Tensor grad_output(output.rows(), output.cols());
    grad_output.fill(1.0f);  // Set all gradients to 1
    nn::Tensor grad_input = layer.backward(grad_output);
    std::cout << "Gradient input from backward pass:" << std::endl;
    grad_input.print();
    
    std::cout << "Linear layer test completed." << std::endl << std::endl;
}

void test_activation_layers() {
    std::cout << "Testing Activation layers..." << std::endl;
    
    // Test Sigmoid
    nn::Sigmoid sigmoid;
    nn::Tensor input({{0.0f, 1.0f, -1.0f}}, {3, 1});
    nn::Tensor sig_output = sigmoid.forward(input);
    std::cout << "Sigmoid input:" << std::endl;
    input.print();
    std::cout << "Sigmoid output:" << std::endl;
    sig_output.print();
    
    // Test backward pass
    nn::Tensor grad_out(sig_output.rows(), sig_output.cols());
    grad_out.fill(1.0f);
    nn::Tensor sig_grad = sigmoid.backward(grad_out);
    std::cout << "Sigmoid gradient:" << std::endl;
    sig_grad.print();
    
    // Test ReLU
    nn::ReLU relu;
    nn::Tensor relu_input({{-1.0f, 0.0f, 1.0f, 2.0f}}, {4, 1});
    nn::Tensor relu_output = relu.forward(relu_input);
    std::cout << "ReLU input:" << std::endl;
    relu_input.print();
    std::cout << "ReLU output:" << std::endl;
    relu_output.print();
    
    // Test backward pass
    nn::Tensor relu_grad_out(relu_output.rows(), relu_output.cols());
    relu_grad_out.fill(1.0f);
    nn::Tensor relu_grad = relu.backward(relu_grad_out);
    std::cout << "ReLU gradient:" << std::endl;
    relu_grad.print();
    
    std::cout << "Activation layers test completed." << std::endl << std::endl;
}

void test_loss_functions() {
    std::cout << "Testing Loss functions..." << std::endl;
    
    // Create prediction and target tensors
    nn::Tensor predictions({{0.7f, 0.2f, 0.1f}}, {1, 3});
    nn::Tensor targets({{1.0f, 0.0f, 0.0f}}, {1, 3});
    
    std::cout << "Predictions:" << std::endl;
    predictions.print();
    std::cout << "Targets:" << std::endl;
    targets.print();
    
    // Test MSE loss
    nn::MSELoss mse_loss;
    float mse = mse_loss.compute_loss(predictions, targets);
    std::cout << "MSE Loss: " << mse << std::endl;
    
    nn::Tensor mse_grad = mse_loss.compute_gradient(predictions, targets);
    std::cout << "MSE Gradient:" << std::endl;
    mse_grad.print();
    
    // Test Cross Entropy loss
    nn::CrossEntropyLoss ce_loss;
    float ce = ce_loss.compute_loss(predictions, targets);
    std::cout << "Cross Entropy Loss: " << ce << std::endl;
    
    nn::Tensor ce_grad = ce_loss.compute_gradient(predictions, targets);
    std::cout << "Cross Entropy Gradient:" << std::endl;
    ce_grad.print();
    
    std::cout << "Loss functions test completed." << std::endl << std::endl;
}

void test_optimizers() {
    std::cout << "Testing Optimizers..." << std::endl;
    
    // Create a simple network for testing
    nn::Network net;
    nn::Linear* layer1 = new nn::Linear(2, 3);
    nn::Linear* layer2 = new nn::Linear(3, 1);
    net.add_layer(layer1);
    net.add_layer(layer2);
    
    // Create test data
    nn::Tensor input({{0.5f, 0.3f}}, {2, 1});
    nn::Tensor target({{0.8f}}, {1, 1});
    
    nn::MSELoss loss_fn;
    
    // Test SGD optimizer
    nn::SGD sgd(0.01f);
    float loss1 = net.train(input, target, loss_fn, sgd);
    std::cout << "SGD - Loss after first training step: " << loss1 << std::endl;
    
    float loss2 = net.train(input, target, loss_fn, sgd);
    std::cout << "SGD - Loss after second training step: " << loss2 << std::endl;
    
    // Test Momentum optimizer
    nn::Momentum momentum(0.01f, 0.9f);
    float loss3 = net.train(input, target, loss_fn, momentum);
    std::cout << "Momentum - Loss after training step: " << loss3 << std::endl;
    
    // Test Adam optimizer
    nn::Adam adam(0.001f);
    float loss4 = net.train(input, target, loss_fn, adam);
    std::cout << "Adam - Loss after training step: " << loss4 << std::endl;
    
    // Clean up
    auto& layers = net.get_layers();
    for (auto* layer : layers) {
        delete layer;
    }
    
    std::cout << "Optimizers test completed." << std::endl << std::endl;
}

void test_network_forward_backward() {
    std::cout << "Testing Network forward/backward propagation..." << std::endl;
    
    // Create a simple network: 2 -> 3 -> 1
    nn::Network net;
    nn::Linear* layer1 = new nn::Linear(2, 3);
    nn::Sigmoid* act1 = new nn::Sigmoid();
    nn::Linear* layer2 = new nn::Linear(3, 1);
    nn::Sigmoid* act2 = new nn::Sigmoid();
    
    net.add_layer(layer1);
    net.add_layer(act1);
    net.add_layer(layer2);
    net.add_layer(act2);
    
    // Create test data
    nn::Tensor input({{0.5f, 0.3f}}, {2, 1});
    nn::Tensor target({{0.8f}}, {1, 1});
    
    // Forward pass
    nn::Tensor output = net.forward(input);
    std::cout << "Network forward pass output:" << std::endl;
    output.print();
    
    // Test training
    nn::MSELoss loss_fn;
    nn::SGD optimizer(0.1f);
    
    float initial_loss = net.train(input, target, loss_fn, optimizer);
    std::cout << "Initial loss: " << initial_loss << std::endl;
    
    // Run a few more iterations to see if loss decreases
    for (int i = 0; i < 5; ++i) {
        float loss = net.train(input, target, loss_fn, optimizer);
        std::cout << "Training step " << (i+1) << " loss: " << loss << std::endl;
    }
    
    // Clean up
    auto& layers = net.get_layers();
    for (auto* layer : layers) {
        delete layer;
    }
    
    std::cout << "Network forward/backward propagation test completed." << std::endl << std::endl;
}

int main() {
    std::cout << "Running Neural Network Library Tests" << std::endl << std::endl;
    
    test_tensor_operations();
    test_linear_layer();
    test_activation_layers();
    test_loss_functions();
    test_optimizers();
    test_network_forward_backward();
    
    std::cout << "All tests completed!" << std::endl;
    
    return 0;
}