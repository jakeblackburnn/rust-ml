// use std::process;
use tensor::Tensor;

fn main() {
    let random = Tensor::random_normal(vec![3, 3], 0.0, 0.1);
    let rand_view = random.view();

    println!("Randomly distributed data:");
    println!("{:?}", rand_view.elements);

    let sample = Tensor::from("example.dat");
    let sample_view = sample.view();

    println!("data from file:");
    println!("{:?}", sample_view.elements);
}
