# Rust Machine Learning

A Rust workspace implementing tensor operations and gradient-based linear regression from scratch. This project builds from fundamental tensor structures to complete machine learning algorithms, with a focus on educational implementation of core mathematical concepts.

## Project Overview

This project demonstrates the implementation of machine learning algorithms using pure Rust, starting from basic tensor operations and building up to gradient descent-based linear regression and beyond.  

**Current Status:** - All tensor operations needed for linear regression are implemented and tested. Moving on to linear regression soon!!!

## Project Structure

This project is organized as a Cargo workspace using git subtrees:

```
rust-ml/
├── Cargo.toml                    # Workspace configuration
├── tensor/                       # Tensor operations crate (git subtree)
│   ├── Cargo.toml               # Tensor crate configuration  
│   ├── src/
│   │   ├── lib.rs               # Core tensor implementation
│   │   ├── error.rs             # Error types and handling
│   │   ├── tests.rs             # Comprehensive test suite
│   │   └── main.rs              # Example usage
│   └── README.md                # Tensor crate documentation
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

### Running Tests

```bash
# Run all workspace tests
cargo test

# Run only tensor crate tests
cargo test -p tensor

```

## Features

### ✅ Completed 
- **Tensor Foundation**: N-dimensional tensors with efficient indexing and memory layout
- **Tensor Constructors**: `zeroes()`, `ones()`, `random_normal()` for initialization
- **TensorView System**: Zero-copy tensor slicing and efficient memory access
- **Matrix Operations**: Matrix multiplication, transpose, dot products
- **Element-wise Arithmetic**: Addition, subtraction, scalar multiplication/division
- **Reduction Operations**: Sum and mean calculations for gradient computation

### 🚧 In Progress 
- **Improved Tensor Errors**: handle new edge cases with informative error messages
- **Linear Regression**: Gradient descent-based implementation

## Technologies Used

- **Language**: Rust (2021 edition)
- **Build System**: Cargo workspace

## License

This project is licensed under the [MIT License](LICENSE).

## Contribution Guidelines

This project is primarily educational and not intended for collaboration. However, if you would like to contribute:

1. **Issues** - Use the [Issues tab](https://github.com/jakeblackburnn/rust-ml/issues) to report issues or make suggestions.
2. **Changes** - Fork the repo, create, commit, and push a new branch (**feature/feature-name**), then open a pull request. 
3. **Contact Me** - jackblackburnn@icloud.com

