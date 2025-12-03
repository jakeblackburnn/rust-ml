# Rust Machine Learning

A Rust workspace implementing machine learning from foundational tensor operations to neural networks with backpropagation. This educational project demonstrates the progression from basic N-dimensional arrays through gradient descent algorithms to deep learning on the Iris dataset.

## Project Structure

```
tensor/              # N-dimensional array operations (matrix multiplication, reductions, arithmetic)
neural-network/      # Multi-layer feedforward network with backpropagation and Iris classification
linear_regression/   # Gradient descent-based regression for continuous predictions
perceptron/          # Binary classification using perceptron learning algorithm
```

See individual crate READMEs for detailed documentation.

## Installation

```bash
git clone https://github.com/jakeblackburnn/rust-ml.git
cd rust-ml && cargo build
```

## Quick Start

### Tensor Operations (Foundation)

```rust
use tensor::Tensor;

let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
let result = a.view().matmul(&b.view())?;  // Matrix multiplication
```

### Neural Network (Iris Classification)

```rust
use neural_network::NeuralNetwork;

// Network: 4 inputs → 64 hidden → 64 hidden → 3 outputs
let mut nn = NeuralNetwork::new(vec![4, 64, 64, 3]);

// Train with validation every 100 epochs
nn.train(&train_features, &train_labels,
         Some((&val_features, &val_labels, 100)),
         0.02, 20000)?;

// Evaluate
let predictions = nn.predict(&test_features)?;
```

### Other Algorithms

The `linear_regression` crate implements gradient descent for continuous predictions, and the `perceptron` crate provides binary classification. See their READMEs for examples.

## Running Examples

```bash
cargo run -p tensor              # Tensor operations demo
cargo run -p neural-network      # Train neural network on Iris dataset
cargo run -p linear_regression   # Linear regression training
cargo run -p perceptron          # Perceptron classification
```

## Features

**Tensor Library:** N-dimensional arrays with stride-based layout, matrix operations (multiplication, transpose), element-wise arithmetic, and reduction operations (sum, mean, MSE). Zero-copy TensorView for efficient memory usage.

**Neural Network:** Multi-layer feedforward architecture with ReLU/Softmax activations, backpropagation with cross-entropy loss, batch training with validation support, and loss history tracking. Includes Iris classification example.

**Other Algorithms:** Gradient descent linear regression with convergence detection, and perceptron binary classification.

## Running Tests

```bash
cargo test              # All workspace tests
cargo test -p tensor    # Specific crate tests
```

## Technologies & License

Pure Rust implementation (2024 edition) with minimal dependencies (csv, rand, serde). Licensed under [MIT License](LICENSE).

## Contributing

Educational project. For issues or suggestions: [Issues tab](https://github.com/jakeblackburnn/rust-ml/issues) or jackblackburnn@icloud.com.
