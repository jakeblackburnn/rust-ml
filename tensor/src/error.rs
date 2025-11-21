use std::fmt;

#[derive(Debug)]
pub enum TensorError {
    RankMismatch {
        provided: usize,
        expected: usize,
    },
    ShapeMismatch {
        provided: Vec<usize>,
        expected: Vec<usize>
    },
    CoordsOutOfBounds {
        rank: usize,
        provided: usize,
        max: usize,
    },
    IncompatibleDimensions {
        k1: usize,
        k2: usize,
    },
    NotMatrixError(usize),
    NotVectorError(usize),
    IncompatibleShapes {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    InvalidLogarithmInput,

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

            TensorError::ShapeMismatch { provided, expected } => {
                write!(
                    f,
                    "Tensor shapes did not match. found (other.shape): {:?}, expected (self.shape): {:?}",
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

            TensorError::IncompatibleDimensions {
                k1,
                k2,
            } => {
                write!(
                    f,
                    "Incompatible inner dimentions: k1 (self.shape[1]): {}, k2 (other.shape[0]): {}",
                    k1, k2
                )
            }

            TensorError::NotMatrixError(rank) => {
                write!(f, "Expected Matrix, found rank {} tensor.", rank)
            }

            TensorError::NotVectorError(rank) => {
                write!(f, "Expected Vector, found rank {} tensor.", rank)
            }

            TensorError::IncompatibleShapes { left, right } => {
                write!(
                    f,
                    "Incompatible shapes for broadcasting: {:?} and {:?}",
                    left, right
                )
            }

            TensorError::InvalidLogarithmInput => {
                write!(f, "Cannot compute logarithm of negative value")
            }
        }
    }
}
