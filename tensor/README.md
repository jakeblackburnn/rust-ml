# Tensor

A Rust library implementing tensor operations and linear algebra for machine learning applications. This project builds from basic tensor structures to machine learning algorithms like linear regression, and ultimately basic multi-layer neural networks.

## Features

- **Tensor Operations**: N-dimensional tensor structure with efficient indexing and bounds checking
- **Matrix Operations**: Matrix multiplication, dot products, element-wise arithmetic (add/subtract), row/column slicing  
- **TensorView System**: Zero-copy tensor views with stride-based memory layout
- **Errors**: Comprehensive error handling with custom error types

## Current Implementation

The library currently provides:

- `Tensor`: Core tensor structure with `Vec<f32>` storage and shape information
- `TensorView`: Memory-efficient views for sub-tensor operations without data copying
- Matrix operations: `matmul()`, `dot()`, `add()`, `sub()`, `row()`, `column()`
- N-dimensional indexing with bounds checking
- Custom error types for dimension mismatches and out-of-bounds access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jakeblackburnn/rust-ml-tensor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd tensor
   ```
3. Build the project:
   ```bash
   cargo build
   ```

## Usage

```rust
use tensor::{Tensor, TensorView};

// Create a 2x3 tensor
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = Tensor::new(data, vec![2, 3]);

// Create a view for efficient operations
let view = TensorView::new(&tensor);

// Access elements
let element = view.get(&[1, 2])?; // Gets element at row 1, column 2

// Matrix operations
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
let view_a = TensorView::new(&a);
let view_b = TensorView::new(&b);
let result = view_a.matmul(&view_b)?;

// Element-wise arithmetic operations
let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
let y = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
let view_x = TensorView::new(&x);
let view_y = TensorView::new(&y);

// Addition: [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0] = [5.0, 7.0, 9.0]
let sum = view_x.add(&view_y)?;

// Subtraction: [4.0, 5.0, 6.0] - [1.0, 2.0, 3.0] = [3.0, 3.0, 3.0]
let diff = view_y.sub(&view_x)?;
```

## Technologies Used

- **Language**: Rust

## Testing

Run tests with:
```bash
cargo test
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contribution Guidelines

This project is not intended for collaboration. However, if you would like to contribute:

1. **Issues** - Use the [Issues tab](https://github.com/jakeblackburnn/rust-tensor/issues) to report issues or make suggestions.
2. **Changes** - Fork the repo, create, commit, and push a new branch (**feature/feature-name**), then open a pull request. 
3. **Contact Me** - jackblackburnn@icloud.com

