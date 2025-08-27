mod error;
use error::TensorError;

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

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..(shape.len() - 1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub struct TensorView<'a> {
    elements: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_vector_element() {
        let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
        assert_eq!(x.get_nd(&[0]).unwrap(), 1.0);
    }

    #[test]
    fn calculate_strides_1d() {
        assert_eq!(calculate_strides(&[3]), vec![1]);
    }

    #[test]
    fn calculate_strides_2d() {
        assert_eq!(calculate_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn calculate_strides_3d() {
        assert_eq!(calculate_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn tensor_view_constructor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = TensorView::new(&tensor);

        assert_eq!(view.shape, vec![2, 3]);
        assert_eq!(view.strides, vec![3, 1]);
        assert_eq!(view.elements.len(), 6);
        assert_eq!(view.elements, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_view_at_2d() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = TensorView::new(&tensor);

        let row0 = view.at(0).unwrap();
        assert_eq!(row0.shape, vec![3]);
        assert_eq!(row0.strides, vec![1]);
        assert_eq!(row0.elements, &[1.0, 2.0, 3.0]);

        let row1 = view.at(1).unwrap();
        assert_eq!(row1.shape, vec![3]);
        assert_eq!(row1.strides, vec![1]);
        assert_eq!(row1.elements, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_view_at_3d() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let view = TensorView::new(&tensor);

        let slice0 = view.at(0).unwrap();
        assert_eq!(slice0.shape, vec![2, 2]);
        assert_eq!(slice0.strides, vec![2, 1]);
        assert_eq!(slice0.elements, &[1.0, 2.0, 3.0, 4.0]);

        let slice1 = view.at(1).unwrap();
        assert_eq!(slice1.shape, vec![2, 2]);
        assert_eq!(slice1.strides, vec![2, 1]);
        assert_eq!(slice1.elements, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn tensor_view_at_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let view = TensorView::new(&tensor);

        let result = view.at(3);
        assert!(result.is_err());

        if let Err(TensorError::CoordsOutOfBounds {
            rank,
            provided,
            max,
        }) = result
        {
            assert_eq!(rank, 1);
            assert_eq!(provided, 3);
            assert_eq!(max, 2);
        }
    }

    #[test]
    fn tensor_view_get_1d() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let view = TensorView::new(&tensor);

        assert_eq!(view.get(&[0]).unwrap(), 1.0);
        assert_eq!(view.get(&[1]).unwrap(), 2.0);
        assert_eq!(view.get(&[2]).unwrap(), 3.0);
    }

    #[test]
    fn tensor_view_get_2d() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = TensorView::new(&tensor);

        assert_eq!(view.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(view.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(view.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(view.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(view.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(view.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn tensor_view_get_3d() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let view = TensorView::new(&tensor);

        assert_eq!(view.get(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(view.get(&[0, 0, 1]).unwrap(), 2.0);
        assert_eq!(view.get(&[0, 1, 0]).unwrap(), 3.0);
        assert_eq!(view.get(&[0, 1, 1]).unwrap(), 4.0);
        assert_eq!(view.get(&[1, 0, 0]).unwrap(), 5.0);
        assert_eq!(view.get(&[1, 0, 1]).unwrap(), 6.0);
        assert_eq!(view.get(&[1, 1, 0]).unwrap(), 7.0);
        assert_eq!(view.get(&[1, 1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn tensor_view_get_rank_mismatch() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::new(&tensor);

        let result = view.get(&[0]);
        assert!(result.is_err());

        if let Err(TensorError::RankMismatch { provided, expected }) = result {
            assert_eq!(provided, 1);
            assert_eq!(expected, 2);
        }

        let result = view.get(&[0, 1, 2]);
        assert!(result.is_err());

        if let Err(TensorError::RankMismatch { provided, expected }) = result {
            assert_eq!(provided, 3);
            assert_eq!(expected, 2);
        }
    }

    #[test]
    fn tensor_view_get_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::new(&tensor);

        let result = view.get(&[2, 0]);
        assert!(result.is_err());

        if let Err(TensorError::CoordsOutOfBounds {
            rank,
            provided,
            max,
        }) = result
        {
            assert_eq!(rank, 1);
            assert_eq!(provided, 2);
            assert_eq!(max, 1);
        }

        let result = view.get(&[0, 3]);
        assert!(result.is_err());

        if let Err(TensorError::CoordsOutOfBounds {
            rank,
            provided,
            max,
        }) = result
        {
            assert_eq!(rank, 2);
            assert_eq!(provided, 3);
            assert_eq!(max, 1);
        }
    }

    #[test]
    fn tensor_view_get_single_element() {
        let tensor = Tensor::new(vec![42.0], vec![1]);
        let view = TensorView::new(&tensor);

        assert_eq!(view.get(&[0]).unwrap(), 42.0);

        let result = view.get(&[1]);
        assert!(result.is_err());
    }
}
