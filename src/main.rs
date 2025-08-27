use std::process;
use tensor::{Tensor, TensorView};

fn main() {
    
    // basic tensor functionality

    let tensor = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);

    println!("attempting to index item from tensor...");
    let x = tensor.get_nd(&[0]).unwrap_or_else(|err| {
        eprintln!("Unrecoverable Tensor Error: {err}");
        process::exit(1);
    });
    println!("successfully indexed item from tensor: {}", x);

    println!("attempting to index item from tensor view...");
    let tensor_view = TensorView::new(&tensor);
    let x = tensor_view.get(&[0]).unwrap_or_else(|err| {
        eprintln!("Unrecoverable Tensor Error: {err}");
        process::exit(1);
    });
    println!("successfully indexed item from tensor view: {}", x);

    // matmul
    
    let matx = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let matx_view = TensorView::new(&matx);

    let other_matx = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let other_matx_view = TensorView::new(&other_matx);

    let result = matx_view.matmul(&other_matx_view).unwrap();
    let result_view = TensorView::new(&result);

    let x = result_view.get(&[0, 0]).unwrap();
    println!("successfully indexed item from matmul tensor: {}", x);

}
