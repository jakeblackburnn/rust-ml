use tensor::{Tensor, TensorError};
use crate::layer::{Layer, ActivationType};
use serde::Serialize;
use csv::Writer;
use std::fs::File;

/// Records training and validation loss for a single epoch
#[derive(Debug, Clone, Serialize)]
pub struct LossRecord {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f32,
}

impl NeuralNetwork {
    /// Creates a new neural network from layer size specifications
    ///
    /// # Arguments
    /// * `layer_sizes` - Array of layer sizes, e.g., [4, 16, 3] creates:
    ///   - Layer 1: 4 inputs → 16 outputs (ReLU)
    ///   - Layer 2: 16 inputs → 3 outputs (Softmax)
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Example
    /// ```
    /// let nn = NeuralNetwork::new(&[4, 16, 3], 0.01);
    /// ```
    pub fn new(layer_sizes: &[usize], learning_rate: f32) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least 2 layers (input + output)");

        let mut layers = Vec::new();

        // Create layers
        for i in 0..layer_sizes.len() - 1 {
            let n_inputs = layer_sizes[i];
            let n_outputs = layer_sizes[i + 1];

            // Use ReLU for hidden layers, Softmax for output layer
            let activation = if i == layer_sizes.len() - 2 {
                ActivationType::Softmax
            } else {
                ActivationType::ReLU
            };

            layers.push(Layer::new(n_inputs, n_outputs, activation));
        }

        NeuralNetwork {
            layers,
            learning_rate,
        }
    }

    /// Forward pass through all layers
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, n_features]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, n_classes]
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {
        let mut output = input.clone();

        // Chain forward passes through all layers
        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }

        Ok(output)
    }

    /// Backward pass through all layers with weight updates
    ///
    /// # Arguments
    /// * `loss_gradient` - Gradient of loss w.r.t. network output
    ///   For softmax + cross-entropy, this should be `y_pred - y_true`
    ///
    /// # Notes
    /// This method both computes gradients and updates weights in one pass
    pub fn backward(&mut self, loss_gradient: &Tensor) -> Result<(), TensorError> {
        let mut grad_output = loss_gradient.clone();

        // Propagate gradients backward through layers
        for layer in self.layers.iter_mut().rev() {
            // Compute gradients for this layer
            let (grad_input, grad_weights, grad_biases) = layer.backward(&grad_output)?;

            // Update weights: w_new = w_old - learning_rate * grad_w
            let weight_update = grad_weights.view().mult(self.learning_rate)?;
            layer.weights = layer.weights.view().sub(&weight_update.view())?;

            // Update biases: b_new = b_old - learning_rate * grad_b
            let bias_update = grad_biases.view().mult(self.learning_rate)?;
            layer.biases = layer.biases.view().sub(&bias_update.view())?;

            // Propagate gradient to previous layer
            grad_output = grad_input;
        }

        Ok(())
    }

    /// Train the network on a dataset
    ///
    /// # Arguments
    /// * `X` - Training data of shape [batch_size, n_features]
    /// * `y_true` - One-hot encoded labels of shape [batch_size, n_classes]
    /// * `epochs` - Number of training iterations
    /// * `x_val` - Optional validation data of shape [batch_size, n_features]
    /// * `y_val` - Optional validation labels of shape [batch_size, n_classes]
    /// * `val_freq` - How often to calculate validation loss (e.g., 100 = every 100 epochs)
    ///
    /// # Returns
    /// Vector of LossRecord containing training and validation losses per epoch
    pub fn train(
        &mut self,
        X: &Tensor,
        y_true: &Tensor,
        epochs: usize,
        x_val: Option<&Tensor>,
        y_val: Option<&Tensor>,
        val_freq: usize,
    ) -> Result<Vec<LossRecord>, TensorError> {
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            // Forward pass
            let y_pred = self.forward(X)?;

            // Compute training loss
            let train_loss = cross_entropy_loss(&y_pred, y_true)?;

            // Compute loss gradient
            let loss_grad = cross_entropy_gradient(&y_pred, y_true)?;

            // Backward pass with weight updates
            self.backward(&loss_grad)?;

            // Compute validation loss if validation data is provided and it's the right epoch
            // This must happen AFTER backward pass to avoid layer cache conflicts
            let val_loss = if epoch % val_freq == 0 && x_val.is_some() && y_val.is_some() {
                let x_val = x_val.unwrap();
                let y_val = y_val.unwrap();
                let y_val_pred = self.predict(x_val)?;
                Some(cross_entropy_loss(&y_val_pred, y_val)?)
            } else {
                None
            };

            // Record loss for this epoch
            loss_history.push(LossRecord {
                epoch,
                train_loss,
                val_loss,
            });

            // Print progress every 100 epochs
            if (epoch + 1) % 100 == 0 {
                match val_loss {
                    Some(vl) => println!(
                        "Epoch {}/{}: train_loss = {:.4}, val_loss = {:.4}",
                        epoch + 1, epochs, train_loss, vl
                    ),
                    None => println!(
                        "Epoch {}/{}: train_loss = {:.4}",
                        epoch + 1, epochs, train_loss
                    ),
                }
            }
        }

        Ok(loss_history)
    }

    /// Make predictions without updating weights
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, n_features]
    ///
    /// # Returns
    /// Prediction tensor of shape [batch_size, n_classes] with class probabilities
    pub fn predict(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {
        self.forward(input)
    }

    /// Saves loss history to a CSV file
    ///
    /// # Arguments
    /// * `loss_history` - Vector of LossRecord containing training metrics
    /// * `file_path` - Path where the CSV file will be written
    ///
    /// # Returns
    /// Result indicating success or error
    ///
    /// # Example
    /// ```
    /// let loss_history = network.train(&x_train, &y_train, 1000, Some(&x_val), Some(&y_val), 100)?;
    /// NeuralNetwork::save_loss_history(&loss_history, "loss_history.csv")?;
    /// ```
    pub fn save_loss_history(
        loss_history: &[LossRecord],
        file_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(file_path)?;
        let mut writer = Writer::from_writer(file);

        // Write all records (serde handles the serialization automatically)
        for record in loss_history {
            writer.serialize(record)?;
        }

        writer.flush()?;
        Ok(())
    }
}

/// Computes cross-entropy loss between predictions and true labels
///
/// # Formula
/// L = -sum(y_true * log(y_pred)) / batch_size
///
/// # TODO
/// Currently returns placeholder value. Requires implementing Tensor::ln()
/// method to compute proper cross-entropy loss.
///
/// # Arguments
/// * `y_pred` - Predicted probabilities of shape [batch_size, n_classes]
/// * `y_true` - One-hot encoded labels of shape [batch_size, n_classes]
pub fn cross_entropy_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<f32, TensorError> {
    // Validate shapes match
    if y_pred.shape != y_true.shape {
        return Err(TensorError::ShapeMismatch {
            provided: y_pred.shape.clone(),
            expected: y_true.shape.clone(),
        });
    }

    // Compute cross-entropy loss: L = -sum(y_true * ln(y_pred)) / batch_size
    // Note: ln() method already includes epsilon (1e-7) for numerical stability

    // Step 1: Compute ln(y_pred) - epsilon already handled in ln()
    let log_pred = y_pred.view().ln()?;

    // Step 2: Element-wise multiply: y_true * ln(y_pred)
    let prod = y_true.view().elem_wise_mult(&log_pred.view())?;

    // Step 3: Sum all elements
    let sum = prod.view().sum()?;

    // Step 4: Negate and divide by batch_size
    let batch_size = if y_pred.shape.len() == 2 {
        y_pred.shape[0] as f32
    } else {
        1.0
    };

    let loss = -sum / batch_size;

    Ok(loss)
}

/// Computes gradient of cross-entropy loss w.r.t. predictions
///
/// # Formula
/// For softmax + cross-entropy, the derivative simplifies to:
/// dL/dy_pred = y_pred - y_true
///
/// # Arguments
/// * `y_pred` - Predicted probabilities of shape [batch_size, n_classes]
/// * `y_true` - One-hot encoded labels of shape [batch_size, n_classes]
///
/// # Returns
/// Gradient tensor of same shape as inputs
pub fn cross_entropy_gradient(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, TensorError> {
    // Validate shapes match
    if y_pred.shape != y_true.shape {
        return Err(TensorError::ShapeMismatch {
            provided: y_pred.shape.clone(),
            expected: y_true.shape.clone(),
        });
    }

    // For softmax + cross-entropy, gradient is simply: y_pred - y_true
    y_pred.view().sub(&y_true.view())
}
