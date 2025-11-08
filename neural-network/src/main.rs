use neural_network::layer::{Layer, ActivationType};
use tensor::Tensor;

fn main() {
    println!("=== Phase 2 Demo: Single Layer Forward Pass ===\n");

    // Create a layer: 4 inputs -> 3 outputs (like final Iris classification layer)
    let mut layer = Layer::new(4, 3, ActivationType::Softmax);
    println!("Created layer: 4 inputs -> 3 outputs with Softmax activation\n");

    // Create fake input: 2 samples, 4 features each
    let input = Tensor::random_normal(vec![2, 4], 0.0, 1.0);
    println!("Input shape: {:?}", input.shape);
    println!("Input data:\n{:?}\n", input);

    // Forward pass
    let output = layer.forward(&input).expect("Forward pass failed");

    println!("Output shape: {:?}", output.shape);
    println!("Output data:\n{:?}\n", output);

    // Verify softmax: each row should sum to 1.0
    println!("=== Verification ===");
    let n_outputs = output.shape[1];
    for sample_idx in 0..2 {
        let mut row_sum = 0.0;
        print!("Sample {} probabilities: [", sample_idx);
        for out_idx in 0..n_outputs {
            let value = output.get_nd(&[sample_idx, out_idx]).expect("Failed to get value");
            row_sum += value;
            print!("{:.4}", value);
            if out_idx < n_outputs - 1 {
                print!(", ");
            }
        }
        println!("] -> sum = {:.6}", row_sum);
    }

    println!("\nPhase 2 Complete! Single layer forward pass is working.");
}
