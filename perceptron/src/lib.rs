use tensor::{Tensor, TensorView, TensorError};

pub struct Perceptron {
    weights: Tensor,
}

impl Perceptron {
    pub fn new(num_weights: usize) -> Self {
        let weights = Tensor::random_normal(vec![num_weights], 0.0, 0.1);
        Perceptron { weights }
    }

    pub fn predict(&self, features: &TensorView) -> Result<i32, TensorError> {
        let weights_view = self.weights.view();
        let hypothesis = weights_view.dot(features)?;
        Ok(if hypothesis > 0.0 { 1 } else { -1 })
    }

    pub fn fit(&mut self, dat_file: &str) -> Result<(), TensorError> {
        let dat_mat = self.load_dat_file(dat_file)?;
        let dat_view = dat_mat.view();

        // Perceptron Learning Algorithm
        let mut misclassified = true;
        while misclassified {
            misclassified = false;
            for i in 0..dat_view.shape[0] {
                let example = dat_view.at(i)?;
                let features = example.slice(0, dat_view.shape[1] - 1)?;
                let target = example.get(&[dat_view.shape[1] - 1])? as i32;

                let prediction = self.predict(&features)?;

                if prediction == target { continue; }

                let weights_view = self.weights.view();

                if target == 1 {
                    self.weights = weights_view.add(&features)?;
                } else {
                    self.weights = weights_view.sub(&features)?;
                }

                misclassified = true;

            }
        }
        
        Ok(())
    }

    pub fn evaluate(&self, dat_file: &str) -> Result<(), TensorError> {
        let dat_mat = self.load_dat_file(dat_file)?;
        let dat_view = dat_mat.view();

        let mut total_correct = 0;

        for i in 0..dat_view.shape[0] {
            let example = dat_view.at(i)?;
            let features = example.slice(0, dat_view.shape[1] - 1)?;
            let target = example.get(&[dat_view.shape[1] - 1])? as i32;

            let prediction = self.predict(&features)?;

            if prediction == target { total_correct += 1 }
        }

        let accuracy = (total_correct as f32) / (dat_view.shape[0] as f32) * 100.0;

        println!("model evaluated {:.2}% accurate", accuracy);
        Ok(())
    }

    fn load_dat_file(&self, dat_file: &str) -> Result<Tensor, TensorError> {
        let raw_data = Tensor::from(dat_file);
        let rows = raw_data.shape[0];
        let original_cols = raw_data.shape[1];
        let new_cols = original_cols + 1;
        
        let mut new_elements = Vec::with_capacity(rows * new_cols);
        
        for row in 0..rows {
            new_elements.push(1.0); // bias term
            for col in 0..original_cols {
                let val = raw_data.get_nd(&[row, col])?;
                new_elements.push(val);
            }
        }
        
        Ok(Tensor::new(new_elements, vec![rows, new_cols]))
    }
}
