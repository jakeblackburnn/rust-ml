# Linear Regression

A Rust implementation of linear regression using gradient descent optimization. This implementation leverages the custom tensor library for efficient matrix operations and provides a robust solution for fitting linear models to continuous data.

## Features

- **Gradient Descent Optimization**: Implements gradient descent algorithm with configurable learning rate and maximum iterations
- **Mean Squared Error Cost Function**: Uses MSE as the loss function with automatic convergence detection
- **Tensor Integration**: Built on top of the custom tensor library for efficient vector and matrix operations
- **Automatic Bias Handling**: Automatically adds bias terms to input features during training and prediction
- **Model Visualization**: Print learned model equations in human-readable format
- **Early Convergence Detection**: Automatically stops training when cost function converges
- **File-Based Data Loading**: Support for loading training data from `.dat` files

## Current Implementation

The library currently provides:

- `LinearRegression`: Core linear regression structure with weights, learning rate, and iteration controls
- `new()`: Initialize model with specified number of features, learning rate, and maximum iterations
- `fit()`: Train the model using gradient descent on data files with automatic convergence detection
- `predict()`: Make predictions on input feature matrices
- `print()`: Display the learned linear equation in readable format (e.g., "model: 2.50 + 3.20x1 - 1.10x2 = y")
- Integration with custom tensor library for all numerical operations
- Automatic bias term addition during data loading

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jakeblackburnn/rust-ml-tensor.git
   ```
2. Navigate to the linear regression directory:
   ```bash
   cd linear_regression
   ```
3. Build the project:
   ```bash
   cargo build
   ```

## Usage

### Basic Linear Regression Training

```rust
use linear_regression::LinearRegression;

// Create a new linear regression model
// Parameters: num_features, learning_rate, max_iterations
let mut model = LinearRegression::new(2, 0.0001, 1000000);

// Print initial random weights
model.print(); // e.g., "model: 45.23 + 12.67x1 = y"

// Train the model on data file
model.fit("train.dat").unwrap();

// Print learned model
model.print(); // e.g., "model: 25000.45 + 15000.30x1 = y"
```

### Making Predictions

```rust
use linear_regression::LinearRegression;
use tensor::Tensor;

let mut model = LinearRegression::new(2, 0.0001, 100000);
// ... train the model ...

// Create feature matrix (bias will be added automatically)
let features = Tensor::new(vec![1.5, 2.0, 2.5], vec![3, 1]); // 3 examples, 1 feature each
let features_view = features.view();

// Make predictions
let predictions = model.predict(&features_view).unwrap();
// predictions now contains the predicted values for each input
```

### Custom Hyperparameters

```rust
use linear_regression::LinearRegression;

// Fine-tune hyperparameters for your specific problem
let mut model = LinearRegression::new(
    3,        // Number of features (excluding bias)
    0.01,     // Learning rate (higher = faster convergence, but may overshoot)
    50000     // Maximum iterations (prevents infinite loops)
);

model.fit("data.dat").unwrap();
```

## Data Format

The linear regression expects data files (`.dat`) in space-separated format:

```
feature1 feature2 ... featureN target
```

Example training data (`train.dat`):
```
1.1 39343.00
1.3 46205.00
1.5 37731.00
2.0 43525.00
2.2 39891.00
2.9 56642.00
3.0 60150.00
```

Where:
- Each row represents one training example
- Features are floating-point numbers (e.g., years of experience)
- Target is the continuous value to predict (e.g., salary)
- Bias terms are automatically added during data loading

For multi-feature regression:
```
feature1 feature2 feature3 target
1.1 2.5 3.0 39343.00
1.3 2.8 3.2 46205.00
1.5 3.1 3.5 37731.00
```

## Algorithm Details

### Gradient Descent Implementation

1. **Initialization**: Weights are initialized with random values from a normal distribution (mean=0, std=100)
2. **Forward Pass**: Compute predictions using `predictions = features × weights`
3. **Cost Calculation**: Compute Mean Squared Error: `MSE = (1/2m) × Σ(predictions - targets)²`
4. **Gradient Calculation**: Compute gradients: `∇ = (2/m) × features^T × (predictions - targets)`
5. **Weight Update**: Update weights: `weights = weights - learning_rate × gradient`
6. **Convergence Check**: Stop if change in cost < 1e-6

### Key Parameters

- **Learning Rate**: Controls step size in gradient descent (typical values: 0.01, 0.001, 0.0001)
- **Max Iterations**: Prevents infinite loops (typical values: 10,000 - 1,000,000)
- **Convergence Threshold**: 1e-6 change in cost function

### Model Equation Output

The `print()` method displays the learned linear equation:
```
model: 25000.45 + 15000.30x1 - 2000.15x2 + 500.75x3 = y
```

This represents: `y = 25000.45 + 15000.30×feature1 - 2000.15×feature2 + 500.75×feature3`

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

## Performance Tips

- **Learning Rate**: Start with 0.01 and decrease if cost increases or oscillates
- **Feature Scaling**: Consider normalizing features if they have vastly different scales
- **Max Iterations**: Increase if model hasn't converged, decrease for faster training
- **Data Size**: More training data generally leads to better generalization

## License

This project is licensed under the [MIT License](LICENSE).

## Contribution Guidelines

This project is not intended for collaboration. However, if you would like to contribute:

1. **Issues** - Use the [Issues tab](https://github.com/jakeblackburnn/rust-tensor/issues) to report issues or make suggestions.
2. **Changes** - Fork the repo, create, commit, and push a new branch (**feature/feature-name**), then open a pull request. 
3. **Contact Me** - jackblackburnn@icloud.com