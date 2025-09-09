# Rust Machine Learning

A comprehensive Rust workspace implementing fundamental machine learning algorithms from scratch. This educational project builds from basic tensor operations to complete machine learning algorithms including linear regression and perceptron classification, with a focus on understanding core mathematical concepts through implementation.

## Project Overview

This project demonstrates the implementation of machine learning algorithms using pure Rust, starting from fundamental tensor operations and building up to gradient descent-based linear regression and perceptron classification. All algorithms are implemented from scratch using custom tensor operations for maximum educational value.

**Current Status:** ✅ **Complete** - All major components implemented: tensor operations, linear regression with gradient descent, and perceptron binary classification.

## Project Structure

This project is organized as a Cargo workspace with three main crates:

```
rust-ml/
├── Cargo.toml                    # Workspace configuration
├── tensor/                       # Core tensor operations library
│   ├── Cargo.toml               # Tensor crate configuration  
│   ├── src/
│   │   ├── lib.rs               # Core tensor implementation
│   │   ├── error.rs             # Error types and handling
│   │   ├── tests.rs             # Comprehensive test suite
│   │   └── main.rs              # Example usage
│   └── README.md                # Tensor crate documentation
├── linear_regression/            # Gradient descent linear regression
│   ├── Cargo.toml               # Linear regression configuration
│   ├── src/
│   │   ├── lib.rs               # Linear regression implementation
│   │   └── main.rs              # Training example
│   ├── train.dat                # Sample training data
│   └── README.md                # Linear regression documentation
├── perceptron/                   # Binary classification perceptron
│   ├── Cargo.toml               # Perceptron crate configuration
│   ├── src/
│   │   ├── lib.rs               # Perceptron algorithm implementation
│   │   └── main.rs              # Classification example
│   ├── train.dat                # Training data (linearly separable)
│   ├── test.dat                 # Test data
│   └── README.md                # Perceptron documentation
└── target/                      # Build artifacts
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jakeblackburnn/rust-ml.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rust-ml
   ```
3. Build the entire workspace:
   ```bash
   cargo build
   ```

## Usage

### Tensor Operations

The foundation tensor library provides efficient N-dimensional arrays and operations:

```rust
use tensor::{Tensor, TensorView};

// Create tensors using constructors
let zeros = Tensor::zeroes(vec![2, 3]);        // 2x3 tensor filled with zeros
let ones = Tensor::ones(vec![2, 3]);           // 2x3 tensor filled with ones
let weights = Tensor::random_normal(vec![2, 3], 0.0, 1.0); // Random initialization

// Matrix operations
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
let result = a.view().matmul(&b.view())?;

// Element-wise operations (essential for gradient descent)
let gradients = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);
let learning_rate = 0.01;
let weight_update = gradients.view().mult(learning_rate)?;
```

### Linear Regression

Gradient descent-based linear regression for continuous predictions:

```rust
use linear_regression::LinearRegression;

// Create model: features=2, learning_rate=0.0001, max_iterations=1000000
let mut model = LinearRegression::new(2, 0.0001, 1000000);

// Train on data file (format: "feature1 feature2 target")
model.fit("train.dat").unwrap();

// Print learned equation (e.g., "model: 25000.45 + 15000.30x1 = y")
model.print();
```

### Perceptron Classification

Binary classification using the perceptron learning algorithm:

```rust
use perceptron::Perceptron;

// Create perceptron for 2D data (automatically adds bias)
let mut model = Perceptron::new(3);

// Train on linearly separable data (format: "feature1 feature2 label")
model.fit("train.dat").unwrap();

// Evaluate accuracy
model.evaluate("test.dat").unwrap(); // Prints accuracy percentage
```

### Running Examples

```bash
# Run tensor examples
cd tensor && cargo run

# Run linear regression training
cd linear_regression && cargo run

# Run perceptron classification
cd perceptron && cargo run
```

### Running Tests

```bash
# Run all workspace tests
cargo test

# Run specific crate tests
cargo test -p tensor
cargo test -p linear_regression
cargo test -p perceptron
```

## Features

### ✅ Completed Algorithms

#### Tensor Operations Library
- **N-Dimensional Tensors**: Efficient indexing, memory layout, and bounds checking
- **Tensor Constructors**: `zeroes()`, `ones()`, `random_normal()` for convenient initialization
- **TensorView System**: Zero-copy tensor views with stride-based memory layout for efficient operations
- **Matrix Operations**: Matrix multiplication, transpose, dot products with dimension validation
- **Element-wise Arithmetic**: Addition, subtraction, scalar multiplication/division
- **Reduction Operations**: Sum, mean, square, and MSE calculations for machine learning
- **Comprehensive Error Handling**: Robust error types for shape mismatches and invalid operations

#### Linear Regression
- **Gradient Descent Optimization**: Configurable learning rate and maximum iterations
- **Mean Squared Error Cost Function**: Automatic convergence detection to prevent overfitting
- **Model Visualization**: Human-readable equation printing (e.g., "model: 2.50 + 3.20x1 = y")
- **File-Based Training**: Support for space-separated `.dat` training files
- **Automatic Bias Handling**: Seamless bias term integration during training and prediction

#### Perceptron Binary Classification
- **Classic Perceptron Learning Algorithm**: Guaranteed convergence on linearly separable data
- **Binary Classification**: Robust classification with -1/+1 label support
- **Model Evaluation**: Built-in accuracy calculation on training and testing datasets
- **Iterative Weight Updates**: Automatic weight adjustment until perfect classification
- **Data File Support**: Training and testing on linearly separable datasets

### 🎯 Educational Focus
- **From-Scratch Implementation**: All algorithms implemented using only standard Rust libraries
- **Mathematical Transparency**: Clear implementation of gradient descent, perceptron learning, and tensor operations
- **Comprehensive Documentation**: Each crate includes detailed README with algorithm explanations

## Technologies Used

- **Language**: Rust (2024 edition)
- **Build System**: Cargo workspace with resolver v3
- **Dependencies**: Pure Rust implementation (no external ML libraries)
- **Architecture**: Modular crate design with shared tensor foundation

## License

This project is licensed under the [MIT License](LICENSE).

## Contribution Guidelines

This project is primarily educational and not intended for collaboration. However, if you would like to contribute:

1. **Issues** - Use the [Issues tab](https://github.com/jakeblackburnn/rust-ml/issues) to report issues or make suggestions.
2. **Changes** - Fork the repo, create, commit, and push a new branch (**feature/feature-name**), then open a pull request. 
3. **Contact Me** - jackblackburnn@icloud.com

