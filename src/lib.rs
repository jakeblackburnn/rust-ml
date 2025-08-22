use std::fmt;

#[derive(Debug)]
pub enum TensorError {
    RankMismatch {
        provided: usize,
        expected: usize,
    },
    CoordsOutOfBounds {
        rank: usize,
        provided: usize,
        max: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::RankMismatch { provided, expected } => {
                write!(
                    f,
                    "Coords do not Match Tensor rank: found: {}, expected: {}",
                    provided, expected
                )
            }

            TensorError::CoordsOutOfBounds {
                rank,
                provided,
                max,
            } => {
                write!(
                    f,
                    "Coords went out of bounds at rank {}: found: {}, max: {}",
                    rank, provided, max
                )
            }
        }
    }
}

pub struct Tensor {
    elements: Vec<f32>,
    shape: Vec<usize>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_vector_element() {
        let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
        assert_eq!(x.get_nd(&[0]).unwrap(), 1.0);
    }
}
