pub mod error;
mod tests;
use error::TensorError;

pub struct Tensor {
    elements: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(elements: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor { elements, shape }
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

    pub fn matmul(&self, other: &TensorView) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::RankMismatch { 
                provided: if self.shape.len() != 2 { self.shape.len() } else { other.shape.len() }, 
                expected: 2 
            });
        }
        
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k1 != k2 {
            // return inner dim mismatch error
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

    pub fn dot(&self, other: &TensorView) -> Result<f32, TensorError> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(TensorError::RankMismatch { 
                provided: if self.shape.len() != 1 { self.shape.len() } else { other.shape.len() }, 
                expected: 1 
            });
        }
        
        if self.shape[0] != other.shape[0] {
            // return vec length mismatch error
        }
        
        let mut sum = 0.0;
        for i in 0..self.shape[0] {
            let a = self.get(&[i])?;
            let b = other.get(&[i])?;
            sum += a * b;
        }
        
        Ok(sum)
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
}

