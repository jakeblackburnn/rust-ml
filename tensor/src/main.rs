// use std::process;
use tensor::Tensor;

fn main() {
    let random = Tensor::random_normal(vec![3, 3], 0.0, 0.1);
    let rand_view = random.view();

    println!("{:?}", rand_view.elements);
}
