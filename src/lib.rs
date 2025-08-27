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

    #[test]
    fn tensor_view_dot_product() {
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor_b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let view_a = TensorView::new(&tensor_a);
        let view_b = TensorView::new(&tensor_b);

        let result = view_a.dot(&view_b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn tensor_view_dot_product_rank_mismatch() {
        let tensor_1d = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor_2d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view_1d = TensorView::new(&tensor_1d);
        let view_2d = TensorView::new(&tensor_2d);

        let result = view_1d.dot(&view_2d);
        assert!(result.is_err());
        
        if let Err(TensorError::RankMismatch { provided, expected }) = result {
            assert_eq!(provided, 2);
            assert_eq!(expected, 1);
        }
    }

    #[test]
    fn tensor_view_dot_product_shape_mismatch() {
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor_b = Tensor::new(vec![4.0, 5.0], vec![2]);
        let view_a = TensorView::new(&tensor_a);
        let view_b = TensorView::new(&tensor_b);

        let result = view_a.dot(&view_b);
        assert!(result.is_err());
        
        // Should return RankMismatch error (used as placeholder for ShapeMismatch)
        if let Err(TensorError::RankMismatch { provided, expected }) = result {
            assert_eq!(provided, 2);
            assert_eq!(expected, 3);
        }
    }

    #[test]
    fn tensor_view_column_extraction() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = TensorView::new(&tensor);

        let col0 = view.column(0).unwrap();
        assert_eq!(col0.shape, vec![2]);
        assert_eq!(col0.get(&[0]).unwrap(), 1.0);
        assert_eq!(col0.get(&[1]).unwrap(), 4.0);

        let col1 = view.column(1).unwrap();
        assert_eq!(col1.shape, vec![2]);
        assert_eq!(col1.get(&[0]).unwrap(), 2.0);
        assert_eq!(col1.get(&[1]).unwrap(), 5.0);

        let col2 = view.column(2).unwrap();
        assert_eq!(col2.shape, vec![2]);
        assert_eq!(col2.get(&[0]).unwrap(), 3.0);
        assert_eq!(col2.get(&[1]).unwrap(), 6.0);
    }

    #[test]
    fn tensor_view_column_rank_mismatch() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let view = TensorView::new(&tensor);

        let result = view.column(0);
        assert!(result.is_err());
        
        if let Err(TensorError::RankMismatch { provided, expected }) = result {
            assert_eq!(provided, 1);
            assert_eq!(expected, 2);
        }
    }

    #[test]
    fn tensor_view_column_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::new(&tensor);

        let result = view.column(2);
        assert!(result.is_err());
        
        if let Err(TensorError::CoordsOutOfBounds { rank, provided, max }) = result {
            assert_eq!(rank, 2);
            assert_eq!(provided, 2);
            assert_eq!(max, 1);
        }
    }

    #[test]
    fn tensor_view_row_extraction() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = TensorView::new(&tensor);

        let row0 = view.row(0).unwrap();
        assert_eq!(row0.shape, vec![3]);
        assert_eq!(row0.get(&[0]).unwrap(), 1.0);
        assert_eq!(row0.get(&[1]).unwrap(), 2.0);
        assert_eq!(row0.get(&[2]).unwrap(), 3.0);

        let row1 = view.row(1).unwrap();
        assert_eq!(row1.shape, vec![3]);
        assert_eq!(row1.get(&[0]).unwrap(), 4.0);
        assert_eq!(row1.get(&[1]).unwrap(), 5.0);
        assert_eq!(row1.get(&[2]).unwrap(), 6.0);
    }

    #[test]
    fn tensor_view_row_rank_mismatch() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let view = TensorView::new(&tensor);

        let result = view.row(0);
        assert!(result.is_err());
        
        if let Err(TensorError::RankMismatch { provided, expected }) = result {
            assert_eq!(provided, 1);
            assert_eq!(expected, 2);
        }
    }

    #[test]
    fn tensor_view_row_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::new(&tensor);

        let result = view.row(2);
        assert!(result.is_err());
        
        if let Err(TensorError::CoordsOutOfBounds { rank, provided, max }) = result {
            assert_eq!(rank, 1);
            assert_eq!(provided, 2);
            assert_eq!(max, 1);
        }
    }

    #[test]
    fn tensor_view_matmul() {
        // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Create a 3x2 matrix: [[7, 8], [9, 10], [11, 12]]
        let tensor_b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        
        let view_a = TensorView::new(&tensor_a);
        let view_b = TensorView::new(&tensor_b);

        let result = view_a.matmul(&view_b).unwrap();
        
        assert_eq!(result.shape, vec![2, 2]);
        
        // Expected result: [[58, 64], [139, 154]]
        // Row 0: [1, 2, 3] · [7, 9, 11] = 58, [1, 2, 3] · [8, 10, 12] = 64
        // Row 1: [4, 5, 6] · [7, 9, 11] = 139, [4, 5, 6] · [8, 10, 12] = 154
        assert_eq!(result.get_nd(&[0, 0]).unwrap(), 58.0);
        assert_eq!(result.get_nd(&[0, 1]).unwrap(), 64.0);
        assert_eq!(result.get_nd(&[1, 0]).unwrap(), 139.0);
        assert_eq!(result.get_nd(&[1, 1]).unwrap(), 154.0);
    }
}
