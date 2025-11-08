use tensor::{Tensor, TensorError};

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Softmax,
}

pub struct Layer {
    weights: Tensor,
    biases: Tensor,
    activation_type: ActivationType,

    // Cached for backward pass (used later in Phase 5)
    last_input: Option<Tensor>,
    last_z: Option<Tensor>,
    last_activation: Option<Tensor>,
}

impl Layer {
    /// Creates a new layer with He initialization for weights
    pub fn new(n_inputs: usize, n_outputs: usize, activation_type: ActivationType) -> Self {
        // He initialization: std = sqrt(2 / n_inputs)
        let std_dev = (2.0 / n_inputs as f32).sqrt();
        let weights = Tensor::random_normal(vec![n_inputs, n_outputs], 0.0, std_dev);
        let biases = Tensor::zeroes(vec![n_outputs]);

        Layer {
            weights,
            biases,
            activation_type,
            last_input: None,
            last_z: None,
            last_activation: None,
        }
    }

    /// Helper function to add bias with broadcasting
    /// Adds 1D bias to each row of 2D matrix
    fn add_bias(&self, matrix: &Tensor) -> Result<Tensor, TensorError> {
        if matrix.shape.len() != 2 {
            return Err(TensorError::NotMatrixError(matrix.shape.len()));
        }

        let (batch_size, n_outputs) = (matrix.shape[0], matrix.shape[1]);

        if self.biases.shape[0] != n_outputs {
            return Err(TensorError::ShapeMismatch {
                provided: self.biases.shape.clone(),
                expected: vec![n_outputs],
            });
        }

        let mut result = Vec::with_capacity(batch_size * n_outputs);

        // Add bias to each row
        for batch_idx in 0..batch_size {
            for out_idx in 0..n_outputs {
                let matrix_val = matrix.get_nd(&[batch_idx, out_idx])?;
                let bias_val = self.biases.get_nd(&[out_idx])?;
                result.push(matrix_val + bias_val);
            }
        }

        Ok(Tensor::new(result, vec![batch_size, n_outputs]))
    }

    /// Forward pass through the layer
    /// Input shape: [batch_size, n_inputs]
    /// Output shape: [batch_size, n_outputs]
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {
        // Cache the input for backward pass
        self.last_input = Some(input.clone());

        // Linear transformation: z = input · weights + biases
        // input: [batch_size, n_inputs]
        // weights: [n_inputs, n_outputs]
        // z: [batch_size, n_outputs]
        let z_temp = input.view().matmul(&self.weights.view())?;

        // Add biases with broadcasting (bias is 1D, z_temp is 2D)
        let z = self.add_bias(&z_temp)?;

        // Cache pre-activation for backward pass
        self.last_z = Some(z.clone());

        // Apply activation function
        let activation = match self.activation_type {
            ActivationType::ReLU => z.view().relu()?,
            ActivationType::Softmax => z.view().softmax()?,
        };

        // Cache post-activation output
        self.last_activation = Some(activation.clone());

        Ok(activation)
    }

    /// Backward pass through the layer
    ///
    /// Computes gradients with respect to inputs, weights, and biases.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient from the next layer (dL/dActivation)
    ///   - For output layer with softmax+cross-entropy: pass (y_pred - y_true)
    ///   - For hidden layers: pass gradient from next layer's backward pass
    ///
    /// # Returns
    /// * `(grad_input, grad_weights, grad_biases)` tuple where:
    ///   - `grad_input`: Gradient w.r.t. input (dL/dInput) - pass to previous layer
    ///   - `grad_weights`: Gradient w.r.t. weights (dL/dWeights) - for parameter updates
    ///   - `grad_biases`: Gradient w.r.t. biases (dL/dBiases) - for parameter updates
    ///
    /// # Note
    /// - ReLU derivative computed explicitly inline (1 if z > 0, else 0)
    /// - Softmax assumes cross-entropy loss (simplified derivative)
    /// - Gradients are averaged over the batch
    pub fn backward(&self, grad_output: &Tensor) -> Result<(Tensor, Tensor, Tensor), TensorError> {
        // Validation: Ensure forward pass was called and values are cached
        let last_input = self.last_input.as_ref()
            .expect("backward() called before forward() - no cached input");
        let last_z = self.last_z.as_ref()
            .expect("backward() called before forward() - no cached pre-activation");
        let last_activation = self.last_activation.as_ref()
            .expect("backward() called before forward() - no cached activation");

        // Validate gradient shape matches activation output shape
        if grad_output.shape != last_activation.shape {
            return Err(TensorError::ShapeMismatch {
                provided: grad_output.shape.clone(),
                expected: last_activation.shape.clone(),
            });
        }

        // Extract batch size for averaging gradients
        let batch_size = last_input.shape[0];

        // Step 1: Compute dL/dZ (gradient w.r.t. pre-activation)
        let grad_z = match self.activation_type {
            ActivationType::ReLU => {
                // ReLU derivative: 1 if z > 0, else 0
                // Compute derivative mask explicitly inline
                let relu_derivative_data: Vec<f32> = last_z.elements
                    .iter()
                    .map(|&z| if z > 0.0 { 1.0 } else { 0.0 })
                    .collect();

                let relu_derivative = Tensor::new(relu_derivative_data, last_z.shape.clone());

                // dL/dZ = dL/dActivation ⊙ activation'(Z)
                grad_output.view().elem_wise_mult(&relu_derivative.view())?
            }
            ActivationType::Softmax => {
                // For softmax + cross-entropy, derivative simplifies to:
                // dL/dZ = y_pred - y_true (assumed already computed in grad_output)
                grad_output.clone()
            }
        };

        // Step 2: Compute dL/dWeights = (input^T · dL/dZ) / batch_size
        // Input shape: [batch_size, n_inputs]
        // Input^T shape: [n_inputs, batch_size]
        // dL/dZ shape: [batch_size, n_outputs]
        // Result shape: [n_inputs, n_outputs]
        let input_transposed = last_input.view().transpose()?;
        let grad_weights_sum = input_transposed.matmul(&grad_z.view())?;
        let grad_weights = grad_weights_sum.view().div(batch_size as f32)?;

        // Step 3: Compute dL/dBiases = sum(dL/dZ, axis=0) / batch_size
        // Sum over batch dimension, then average
        // dL/dZ shape: [batch_size, n_outputs]
        // Result shape: [n_outputs]
        let grad_biases_sum = grad_z.view().sum_axis(0)?;
        let grad_biases = grad_biases_sum.view().div(batch_size as f32)?;

        // Step 4: Compute dL/dInput = dL/dZ · weights^T (for propagation to previous layer)
        // dL/dZ shape: [batch_size, n_outputs]
        // Weights^T shape: [n_outputs, n_inputs]
        // Result shape: [batch_size, n_inputs]
        let weights_transposed = self.weights.view().transpose()?;
        let grad_input = grad_z.view().matmul(&weights_transposed)?;

        Ok((grad_input, grad_weights, grad_biases))
    }

    // TODO: Future optimization - Arc-based zero-copy forward pass
    //
    // This method will eliminate tensor cloning during caching by using Arc (atomic reference counting).
    // Performance benefit: ~3x reduction in memory allocation per layer forward pass.
    //
    // Migration path:
    // 1. Change Layer fields to Option<Arc<Tensor>>
    // 2. Accept Arc<Tensor> parameter instead of &Tensor
    // 3. Use Arc::clone() which just increments refcount (no data copy)
    // 4. Return Arc<Tensor> for zero-copy chaining between layers
    //
    // See docs/performance_optimization.md for full implementation details
    //
    // pub fn forward_arc(&mut self, input: Arc<Tensor>) -> Result<Arc<Tensor>, TensorError> {
    //     self.last_input = Some(Arc::clone(&input));
    //
    //     let z = Arc::new(
    //         input.view()
    //             .matmul(&self.weights.view())?
    //             .view()
    //             .add(&self.biases.view())?
    //     );
    //     self.last_z = Some(Arc::clone(&z));
    //
    //     let activation = Arc::new(match self.activation_type {
    //         ActivationType::ReLU => z.view().relu()?,
    //         ActivationType::Softmax => z.view().softmax()?,
    //     });
    //
    //     self.last_activation = Some(Arc::clone(&activation));
    //     Ok(activation)
    // }
}
