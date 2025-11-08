mod tests;
mod error;
pub use error::TensorError;

use std::fs;
use rand::prelude::*;
use rand_distr::Normal;

#[derive(Clone, Debug)]
pub struct Tensor {
    elements: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(elements: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor { elements, shape }
    }

    pub fn zeroes(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        let elements = vec![0.0; len];

        Tensor { elements, shape }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        let elements = vec![1.0; len];

        Tensor { elements, shape }
    }

    pub fn random_normal(shape: Vec<usize>, mean: f32, std_dev: f32) -> Self {
        let len = shape.iter().product();
        let mut elements = vec![0.0; len];

        let normal = Normal::new(mean, std_dev).unwrap();
        let mut rng = thread_rng();

        for i in 0..len {
            elements[i] = rng.sample(normal);
        }

        Tensor { elements, shape }
    }

    pub fn from(file_path: &str) -> Self {
        // TODO: Add error handling for file not found
        let content = fs::read_to_string(file_path).unwrap();
        
        let lines: Vec<&str> = content.trim().split('\n').collect();
        // TODO: Add error handling for empty file
        let rows = lines.len();
        
        // Parse first line to determine columns
        let first_line_values: Vec<&str> = lines[0].split_whitespace().collect();
        let cols = first_line_values.len();
        
        let mut elements = Vec::new();
        
        for line in lines {
            let values: Vec<&str> = line.split_whitespace().collect();
            // TODO: Add error handling for inconsistent row lengths
            for value_str in values {
                // TODO: Add error handling for parse failures
                let value: f32 = value_str.parse().unwrap();
                elements.push(value);
            }
        }
        
        Tensor {
            elements,
            shape: vec![rows, cols],
        }
    }

    pub fn view(&self) -> TensorView {
        TensorView::new(&self)
    }

    pub fn get_nd(&self, coords: &[usize]) -> Result<f32, TensorError> {
        // TODO: check that coords are sensical
        if self.shape.len() != coords.len() {
            return Err(TensorError::RankMismatch {
                provided: coords.len(),
                expected: self.shape.len(),
            });
        }

        let mut idx = 0;
        let mut stride = 1;

        for rank in (0..coords.len()).rev() {
            // iterate over coordinates from outer -> inner ranks
            // (row major)

            let rank_len = self.shape[rank]; // current rank size
            let rank_idx = coords[rank]; // coordinate idx along current rank

            // check for out of bounds error
            if rank_idx >= rank_len {
                return Err(TensorError::CoordsOutOfBounds {
                    rank: rank + 1,
                    provided: rank_idx,
                    max: rank_len - 1,
                });
            }

            idx += stride * rank_idx;
            stride *= rank_len;
        }

        Ok(self.elements[idx])
    }
}

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..(shape.len() - 1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub struct TensorView<'a> {
    pub elements: &'a [f32],
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<'a> TensorView<'a> {
    pub fn new(t: &'a Tensor) -> TensorView<'a> {
        let elements = &t.elements;
        let shape = t.shape.clone();
        let strides = calculate_strides(&t.shape);

        TensorView {
            elements,
            shape,
            strides,
        }
    }

    pub fn get(&self, coords: &[usize]) -> Result<f32, TensorError> {
        if self.shape.len() != coords.len() {
            return Err(TensorError::RankMismatch {
                provided: coords.len(),
                expected: self.shape.len(),
            });
        }

        let mut idx = 0;

        for rank in (0..coords.len()).rev() {
            let rank_idx = coords[rank];
            let rank_len = self.shape[rank];

            if rank_idx >= rank_len {
                return Err(TensorError::CoordsOutOfBounds {
                    rank: rank + 1,
                    provided: rank_idx,
                    max: rank_len - 1,
                });
            }

            idx += self.strides[rank] * rank_idx;
        }
        Ok(self.elements[idx])
    }

    pub fn add(&self, other: &TensorView) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err( TensorError::ShapeMismatch {
                provided: other.shape.clone(),
                expected: self.shape.clone(),
            });
       }

        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];
        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let self_elem = self.get(&coords)?;
            let other_elem = other.get(&coords)?;
            results[i] = self_elem + other_elem;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok( Tensor::new(results, self.shape.clone()))
    }

    pub fn sub(&self, other: &TensorView) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err( TensorError::ShapeMismatch {
                provided: other.shape.clone(),
                expected: self.shape.clone(),
            });
        }

        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];
        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let self_elem = self.get(&coords)?;
            let other_elem = other.get(&coords)?;
            results[i] = self_elem - other_elem;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }


        Ok( Tensor::new(results, self.shape.clone()))
    }

    pub fn elem_wise_mult(&self, other: &TensorView) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err( TensorError::ShapeMismatch {
                provided: other.shape.clone(),
                expected: self.shape.clone(),
            });
        }

        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];
        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let self_elem = self.get(&coords)?;
            let other_elem = other.get(&coords)?;
            results[i] = self_elem * other_elem;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok( Tensor::new(results, self.shape.clone()))
    }

    pub fn mult(&self, factor: f32) -> Result<Tensor, TensorError> {
        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];

        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let elem = self.get(&coords)?;
            results[i] = elem * factor;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok( Tensor::new(results, self.shape.clone()) )
    }
    pub fn div(&self, denominator: f32) -> Result<Tensor, TensorError> {
        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];

        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let elem = self.get(&coords)?;
            results[i] = elem / denominator;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok( Tensor::new(results, self.shape.clone()) )
    }

    pub fn square(&self) -> Result<Tensor, TensorError> {
        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];

        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let elem = self.get(&coords)?;
            results[i] = elem * elem;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok( Tensor::new(results, self.shape.clone()) )
    }

    pub fn relu(&self) -> Result<Tensor, TensorError> {
        let len = self.shape.iter().product();
        let mut results = vec![0.0; len];

        let mut coords = vec![0; self.shape.len()];

        for i in 0..len {
            let elem = self.get(&coords)?;
            results[i] = elem.max(0.0);

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok( Tensor::new(results, self.shape.clone()) )
    }

    pub fn softmax(&self) -> Result<Tensor, TensorError> {
        // Normalize shape: 1D [n] becomes 2D [1, n]
        let (batch_size, num_features) = match self.shape.len() {
            1 => (1, self.shape[0]),
            2 => (self.shape[0], self.shape[1]),
            _ => return Err(TensorError::NotVectorError(self.shape.len())),
        };

        if num_features == 0 {
            return Ok(Tensor::new(vec![], self.shape.clone()));
        }

        let mut results = vec![0.0; batch_size * num_features];

        for batch_idx in 0..batch_size {
            let offset = batch_idx * num_features;
            let row = &self.elements[offset..offset + num_features];
            let out = &mut results[offset..offset + num_features];

            // Step 1: Find max for numerical stability
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Step 2: Compute exp(x - max) and sum
            let sum_exp: f32 = row.iter()
                .enumerate()
                .map(|(j, &x)| {
                    let exp_val = (x - max_val).exp();
                    out[j] = exp_val;
                    exp_val
                })
                .sum();

            // Step 3: Normalize with edge case handling
            if sum_exp > 0.0 && sum_exp.is_finite() {
                out.iter_mut().for_each(|x| *x /= sum_exp);
            } else {
                // Edge case: if sum is zero or non-finite, return uniform distribution
                out.fill(1.0 / num_features as f32);
            }
        }

        Ok(Tensor::new(results, self.shape.clone()))
    }

    pub fn sum(&self) -> Result<f32, TensorError> {
        let len = self.shape.iter().product();

        let mut result = 0.0;
        let mut coords = vec![0; self.shape.len()];

        for _i in 0..len {
            let elem = self.get(&coords)?;
            result += elem;

            // increment coords
            for j in ( 0..coords.len() ).rev() {
                coords[j] += 1;
                if coords[j] < self.shape[j] {
                    break;
                }
                coords[j] = 0;
            }
        }

        Ok(result)
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Tensor, TensorError> {
        // Validate axis is in range
        if axis >= self.shape.len() {
            return Err(TensorError::CoordsOutOfBounds {
                rank: axis + 1,
                provided: axis,
                max: self.shape.len() - 1,
            });
        }

        // Calculate result shape by removing the axis dimension
        let mut result_shape = Vec::new();
        for (i, &dim) in self.shape.iter().enumerate() {
            if i != axis {
                result_shape.push(dim);
            }
        }

        // Handle edge case: if result is scalar (all dimensions summed)
        if result_shape.is_empty() {
            result_shape.push(1);
        }

        // Calculate result size
        let result_len: usize = result_shape.iter().product();
        let mut result_elements = vec![0.0; result_len];

        // Iterate through all positions in the result
        let mut result_coords = vec![0; if result_shape.len() == 1 && result_shape[0] == 1 { 0 } else { result_shape.len() }];

        for result_idx in 0..result_len {
            // Build full coordinates by inserting axis position
            let mut full_coords = Vec::new();
            let mut result_coord_idx = 0;
            for i in 0..self.shape.len() {
                if i == axis {
                    full_coords.push(0); // Placeholder, will iterate
                } else {
                    if result_coord_idx < result_coords.len() {
                        full_coords.push(result_coords[result_coord_idx]);
                        result_coord_idx += 1;
                    }
                }
            }

            // Sum along the axis dimension
            let mut sum = 0.0;
            for axis_val in 0..self.shape[axis] {
                full_coords[axis] = axis_val;
                sum += self.get(&full_coords)?;
            }

            result_elements[result_idx] = sum;

            // Increment result coords
            if !result_coords.is_empty() {
                for j in (0..result_coords.len()).rev() {
                    result_coords[j] += 1;
                    if result_coords[j] < result_shape[j] {
                        break;
                    }
                    result_coords[j] = 0;
                }
            }
        }

        Ok(Tensor::new(result_elements, result_shape))
    }

    pub fn mean(&self) -> Result<f32, TensorError> {
        let sum = self.sum()?;
        let len: usize = self.shape.iter().product();
        let mean = sum / (len as f32);
        Ok(mean)
    }

    pub fn mse(&self, other: &TensorView) -> Result<f32, TensorError> {
        if self.shape != other.shape {
            return Err( TensorError::ShapeMismatch {
                provided: other.shape.clone(),
                expected: self.shape.clone(),
            });
        }

        let errors = self.sub(other)?;
        let squared_errors = errors.view().square()?;
        let mse = squared_errors.view().mean()?;
        Ok(mse)
    }
 

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 {
            return Err( TensorError::NotMatrixError(self.shape.len()) );
        }

        let (m, n) = ( self.shape[0], self.shape[1] );

        let mut result = vec![0.0; n * m];
        let result_shape = vec![n, m];

        for i in 0..m {
            for j in 0..n {
                let item = self.get(&[i, j])?;
                let new_idx = j * m + i;
                result[new_idx] = item;
            }
        }

        Ok( Tensor::new(result, result_shape) )
    }

    pub fn dot(&self, other: &TensorView) -> Result<f32, TensorError> {

        if self.shape.len() != 1 {
            return Err( TensorError::NotVectorError(self.shape.len()) );
        }
        
        if self.shape != other.shape {
            return Err( TensorError::ShapeMismatch {
                provided: other.shape.clone(),
                expected: self.shape.clone(),
            });
        }
        
        let mut sum = 0.0;
        for i in 0..self.shape[0] {
            let a = self.get(&[i])?;
            let b = other.get(&[i])?;
            sum += a * b;
        }
        
        Ok(sum)
    }

    pub fn matmul(&self, other: &TensorView) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err( TensorError::NotMatrixError( 
                if self.shape.len() != 2 { self.shape.len() }
                else { other.shape.len() }
            ));
        }
        
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k1 != k2 {
            return Err( TensorError::IncompatibleDimensions {
                k1: k1,
                k2: k2,
            });
        }

        let mut result = vec![0.0; m * n];
        
        for i in 0..m {
            let row = self.row(i)?;
            for j in 0..n {
                let col = other.column(j)?;
                result[i * n + j] = row.dot(&col)?;
            }
        }
        
        Ok(Tensor::new(result, vec![m, n]))
    }


    pub fn at(&self, idx: usize) -> Result<TensorView, TensorError> {
        if idx >= self.shape[0] {
            return Err(TensorError::CoordsOutOfBounds {
                rank: 1,
                provided: idx,
                max: self.shape[0] - 1,
            });
        }

        let offset = idx * self.strides[0];
        let new_shape = self.shape[1..].to_vec();
        let new_strides = self.strides[1..].to_vec();

        Ok(TensorView {
            elements: &self.elements[offset..offset + self.strides[0]],
            shape: new_shape,
            strides: new_strides,
        })
    }


    pub fn row(&self, row_idx: usize) -> Result<TensorView, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::RankMismatch {
                provided: self.shape.len(),
                expected: 2,
            });
        }
        
        self.at(row_idx)
    }

    pub fn column(&self, col_idx: usize) -> Result<TensorView, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::RankMismatch { 
                provided: self.shape.len(), 
                expected: 2 
            });
        }
        
        if col_idx >= self.shape[1] {
            return Err(TensorError::CoordsOutOfBounds {
                rank: 2,
                provided: col_idx,
                max: self.shape[1] - 1,
            });
        }

        let offset = col_idx;
        let rows = self.shape[0];
        let row_stride = self.strides[0];
        
        Ok(TensorView {
            elements: &self.elements[offset..],
            shape: vec![rows],
            strides: vec![row_stride],  // Jump by full row width to get next element in column
        })
    }

    pub fn slice(&self, start: usize, end: usize) -> Result<TensorView, TensorError> {
        if self.shape.is_empty() {
            return Err(TensorError::RankMismatch {
                provided: 0,
                expected: 1,
            });
        }

        let last_dim = self.shape.len() - 1;
        let dim_size = self.shape[last_dim];
        
        if start >= dim_size || end > dim_size || start >= end {
            return Err(TensorError::CoordsOutOfBounds {
                rank: last_dim + 1,
                provided: if start >= dim_size { start } else { end - 1 },
                max: dim_size - 1,
            });
        }

        let slice_size = end - start;
        
        // For 1D: simple slice
        if self.shape.len() == 1 {
            let offset = start;
            return Ok(TensorView {
                elements: &self.elements[offset..offset + slice_size],
                shape: vec![slice_size],
                strides: vec![1],
            });
        }
        
        // For 2D+: slice along last dimension
        let mut new_shape = self.shape.clone();
        new_shape[last_dim] = slice_size;
        
        let offset = start;
        
        Ok(TensorView {
            elements: &self.elements[offset..],
            shape: new_shape,
            strides: self.strides.clone(),
        })
    }
}

