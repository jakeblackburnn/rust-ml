use tensor::{Tensor, TensorError};

    // TODO: Future optimization - Arc-based zero-copy fwd/back pass
    // See docs/performance_optimization.md for full implementation details

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Softmax,
}

pub struct Layer {
    weights: Tensor,
    biases: Tensor,
    activation_type: ActivationType,

    // cache for backprop
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

    /// Forward pass through the layer
    /// Input shape: [batch_size, n_inputs]
    /// Output shape: [batch_size, n_outputs]
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {

        // Cache the input for backward pass
        self.last_input = Some(input.clone()); // TODO: switch to arcs for performance / zero-copy 

        // Linear transformation: z = input · weights + biases
        // input: [batch_size, n_inputs]
        // weights: [n_inputs, n_outputs]
        // z: [batch_size, n_outputs]
        let z_temp = input.view().matmul(&self.weights.view())?;

        // Add biases with broadcasting (bias is 1D, z_temp is 2D)
        let z = z_temp.view().broadcast_add(&self.biases.view())?;

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
    /// - ReLU derivative computed explicitly inline (1 if z > 0, else 0)
    /// - Softmax assumes cross-entropy loss (simplified derivative)
    /// - Gradients are averaged over the batch
    pub fn backward(&self, grad_output: &Tensor) -> Result<(Tensor, Tensor, Tensor), TensorError> {

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

        // 1: Compute dL/dZ (gradient w.r.t. pre-activation)
        let grad_z = match self.activation_type {

            ActivationType::ReLU => {
                // ReLU derivative: 1 if z > 0, else 0
                // Compute derivative mask explicitly inline
                let relu_derivative_data: Vec<f32> = last_z.view().elements
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

        // 2: Compute dL/dWeights = (input^T · dL/dZ) / batch_size
        // Input shape: [batch_size, n_inputs]
        // dL/dZ shape: [batch_size, n_outputs]
        // Result shape: [n_inputs, n_outputs]
        let input_transposed = last_input.view().transpose()?;
        let grad_weights_sum = input_transposed.view().matmul(&grad_z.view())?;
        let grad_weights = grad_weights_sum.view().div(batch_size as f32)?;

        // 3: Compute dL/dBiases = sum(dL/dZ, axis=0) / batch_size
        // dL/dZ shape: [batch_size, n_outputs]
        // Result shape: [n_outputs]
        let grad_biases_sum = grad_z.view().sum_axis(0)?;
        let grad_biases = grad_biases_sum.view().div(batch_size as f32)?;

        // 4: Compute dL/dInput = dL/dZ · weights^T (for propagation to previous layer)
        // dL/dZ shape: [batch_size, n_outputs]
        // Weights^T shape: [n_outputs, n_inputs]
        // Result shape: [batch_size, n_inputs]
        let weights_transposed = self.weights.view().transpose()?;
        let grad_input = grad_z.view().matmul(&weights_transposed.view())?;

        Ok((grad_input, grad_weights, grad_biases))
    }

}
