use tensor::{Tensor, TensorView};

pub struct LinearRegression {
    weights: Option<Tensor>,
    lr: f32,
    max_iterations: usize,
    done_training: bool,
}

impl LinearRegression {

    pub fn new(lr: f32, max_iterations: usize) -> Self {
        LinearRegression { weights: None, lr, max_iterations, done_training: false }
    }

    pub fn fit(train_data: Tensor) {
    }

    fn predict(X: Tensor) {
    }
}
