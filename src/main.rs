use tensor::Tensor;
use std::process;

fn main() {
    let tensor = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);

    println!("attempting to index item from tensor...");
    let x = tensor.get_nd(&[0]).unwrap_or_else(|err| {
        eprintln!("Unrecoverable Tensor Error: {err}");
        process::exit(1);
    });
    println!("successfully indexed item from Tensor: {}", x);

    println!("attempting to index item from tensor...");
    let x = tensor.get_nd(&[0, 0]).unwrap_or_else(|err| {
        eprintln!("Unrecoverable Tensor Error: {err}");
        process::exit(1);
    });
    println!("successfully indexed item from Tensor: {}", x);
}
