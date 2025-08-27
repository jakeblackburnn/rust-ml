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

impl std::error::Error for TensorError {}

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
