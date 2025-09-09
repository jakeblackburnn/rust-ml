# Perceptron

A Rust implementation of the classic perceptron algorithm for binary classification of linearly separable data. This implementation leverages the custom tensor library for efficient matrix operations and numerical computations.

## Features

- **Binary Classification**: Single-layer perceptron for linearly separable binary classification problems
- **Perceptron Learning Algorithm**: Classic iterative weight update algorithm that guarantees convergence on linearly separable data
- **Tensor Integration**: Built on top of the custom tensor library for efficient vector and matrix operations
- **File-Based Data Loading**: Support for loading training and testing data from `.dat` files
- **Automatic Bias Handling**: Automatically adds bias terms to input features during training and prediction
- **Model Evaluation**: Built-in accuracy evaluation on training and testing datasets

## Current Implementation

The library currently provides:

- `Perceptron`: Core perceptron structure with weight vector and learning capabilities
- `new()`: Initialize perceptron with random weights for specified number of features
- `fit()`: Train the perceptron using the classic perceptron learning algorithm on data files
- `predict()`: Make binary predictions (-1 or 1) on input feature vectors
- `evaluate()`: Calculate and display accuracy on training or testing datasets
- Automatic bias term addition to input features during data loading
- Integration with custom tensor library for all numerical operations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jakeblackburnn/rust-ml-tensor.git
   ```
2. Navigate to the perceptron directory:
   ```bash
   cd perceptron
   ```
3. Build the project:
   ```bash
   cargo build
   ```

## Usage

### Basic Perceptron Training and Evaluation

```rust
use perceptron::Perceptron;

// Create a new perceptron for 2D data (will add bias automatically)
let mut model = Perceptron::new(3); // 3 weights: bias + 2 features

// Train the model on training data
model.fit("train.dat").unwrap();

// Evaluate on training data
model.evaluate("train.dat").unwrap();

// Evaluate on test data
model.evaluate("test.dat").unwrap();
```

### Making Predictions

```rust
use perceptron::Perceptron;
use tensor::Tensor;

let model = Perceptron::new(3);
// ... train the model ...

// Create feature vector (bias will be added automatically during file loading)
let features = Tensor::new(vec![0.5, 0.3], vec![2]);
let features_view = features.view();

// Make prediction (-1 or 1)
let prediction = model.predict(&features_view).unwrap();
println!("Prediction: {}", prediction);
```

## Data Format

The perceptron expects data files (`.dat`) in space-separated format:

```
feature1 feature2 ... featureN label
```

Example training data (`train.dat`):
```
0.5496 0.4353 1.0
0.4204 0.3303 1.0
0.2046 0.6193 1.0
0.2997 0.2668 1.0
0.6211 0.5291 1.0
0.7853 0.2156 -1.0
0.9234 0.4567 -1.0
0.8765 0.6789 -1.0
```

Where:
- Each row represents one training example
- Features are floating-point numbers
- Labels are binary: `1.0` for positive class, `-1.0` for negative class
- Bias terms are automatically added during data loading

## Algorithm Details

The implementation uses the classic perceptron learning algorithm:

1. **Initialization**: Weights are initialized with small random values from a normal distribution
2. **Training Loop**: For each misclassified example:
   - If target is +1 and prediction is -1: `weights = weights + features`
   - If target is -1 and prediction is +1: `weights = weights - features`
3. **Convergence**: Algorithm continues until all training examples are correctly classified
4. **Prediction**: Uses sign of dot product between weights and features

The algorithm is guaranteed to converge on linearly separable datasets.

## Technologies Used

- **Language**: Rust
- **Dependencies**: Custom tensor library for numerical operations

## Testing

Run the example with provided data:
```bash
cargo run
```

Run unit tests:
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