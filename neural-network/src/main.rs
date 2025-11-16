use neural_network::layer::{Layer, ActivationType};
use tensor::Tensor;

fn main() {
    let mut layer = Layer::new(4, 3, ActivationType::Softmax);
    println!("Created layer: 4 inputs -> 3 outputs with Softmax activation\n");

    let input = Tensor::random_normal(vec![2, 4], 0.0, 1.0);
    println!("Input shape: {:?}", input.shape);
    println!("Input data:\n{:?}\n", input);

    let output = layer.forward(&input).expect("Forward pass failed");

    println!("Output shape: {:?}", output.shape);
    println!("Output data:\n{:?}\n", output);
}
