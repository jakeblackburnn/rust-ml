use tensor::{Tensor, TensorView, TensorError};

pub struct LinearRegression {
    weights: Tensor,
    lr: f32,
    max_iterations: usize,
}

impl LinearRegression {

    pub fn new(num_weights: usize, lr: f32, max_iterations: usize) -> Self {
        let weights = Tensor::random_normal(vec![num_weights, 1], 0.0, 100.0);
        LinearRegression { weights, lr, max_iterations }
    }

    pub fn fit(&mut self, dat_file: &str) -> Result<(), TensorError> {

        let dat_mat = self.load_dat_file(dat_file)?;
        let dat_view = dat_mat.view();

        let features = dat_view.slice(0, dat_view.shape[1] - 1)?;
        let targets = dat_view.slice(dat_view.shape[1] - 1, dat_view.shape[1])?; 

        let mut prev_cost = f32::INFINITY;
        for _i in 0..self.max_iterations {
            let predictions = self.predict(&features)?;

            // convergence check
            let cost = predictions.view().mse(&targets)?;
            if (cost - prev_cost).abs() < 1e-6 { break; } // converged
            prev_cost = cost;

            // gradient descent
            let errors = predictions.view().sub(&targets)?;
            let transpose_feats = features.transpose()?;
            let gradient = transpose_feats.view().matmul(&errors.view())?;
            let gradient = gradient.view().mult(2.0 / dat_view.shape[0] as f32)?;

            let update = gradient.view().mult(self.lr)?;

            self.weights = self.weights.view().sub(&update.view())?;
        }
        


        Ok(())
    }

    pub fn predict(&self, dat_view: &TensorView) -> Result<Tensor, TensorError> {
        let weights_view = self.weights.view();
        let predictions_vec = dat_view.matmul(&weights_view)?;
        Ok( predictions_vec )
    }

    pub fn print(&self) {
        let weights_view = self.weights.view();
        
        // Extract bias (first weight)
        let bias = weights_view.get(&[0, 0]).unwrap_or(0.0);
        
        // Start building the equation
        let mut equation = format!("model: {:.2}", bias);
        
        // Add feature weights
        for i in 1..weights_view.shape[0] {
            let weight = weights_view.get(&[i, 0]).unwrap_or(0.0);
            
            if weight >= 0.0 {
                equation.push_str(&format!(" + {:.2}x{}", weight, i));
            } else {
                equation.push_str(&format!(" - {:.2}x{}", weight.abs(), i));
            }
        }
        
        equation.push_str(" = y");
        println!("{}", equation);
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
