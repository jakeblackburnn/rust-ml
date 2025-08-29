#[cfg(test)]
use super::*;

#[test]
fn get_vector_element() {
    let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    assert_eq!(x.get_nd(&[0]).unwrap(), 1.0);
}

#[test]
fn transpose_matx() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_2x2_square() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(yview.shape, vec![2, 2]);
}

#[test]
fn transpose_3x3_square() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    assert_eq!(yview.shape, vec![3, 3]);
}

#[test]
fn transpose_1x4_rectangular() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(yview.shape, vec![4, 1]);
}

#[test]
fn transpose_4x1_rectangular() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(yview.shape, vec![1, 4]);
}

#[test]
fn transpose_2x5_rectangular() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 5]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]);
    assert_eq!(yview.shape, vec![5, 2]);
}

#[test]
fn transpose_1x1_single_element() {
    let x = Tensor::new(vec![42.0], vec![1, 1]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![42.0]);
    assert_eq!(yview.shape, vec![1, 1]);
}

#[test]
fn transpose_double_transpose_identity() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);
    let z = yview.transpose().unwrap();
    let zview = TensorView::new(&z);

    assert_eq!(zview.elements, xview.elements);
    assert_eq!(zview.shape, xview.shape);
}

#[test]
fn transpose_element_position_mapping() {
    let x = Tensor::new(vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0], vec![3, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(xview.get(&[0, 0]).unwrap(), yview.get(&[0, 0]).unwrap()); // 11
    assert_eq!(xview.get(&[0, 1]).unwrap(), yview.get(&[1, 0]).unwrap()); // 12
    assert_eq!(xview.get(&[0, 2]).unwrap(), yview.get(&[2, 0]).unwrap()); // 13
    assert_eq!(xview.get(&[1, 0]).unwrap(), yview.get(&[0, 1]).unwrap()); // 21
    assert_eq!(xview.get(&[1, 1]).unwrap(), yview.get(&[1, 1]).unwrap()); // 22
    assert_eq!(xview.get(&[1, 2]).unwrap(), yview.get(&[2, 1]).unwrap()); // 23
    assert_eq!(xview.get(&[2, 0]).unwrap(), yview.get(&[0, 2]).unwrap()); // 31
    assert_eq!(xview.get(&[2, 1]).unwrap(), yview.get(&[1, 2]).unwrap()); // 32
    assert_eq!(xview.get(&[2, 2]).unwrap(), yview.get(&[2, 2]).unwrap()); // 33
}

#[test]
fn transpose_shape_transformation() {
    let matrices = vec![
        (vec![2, 3], vec![3, 2]),
        (vec![1, 5], vec![5, 1]), 
        (vec![4, 1], vec![1, 4]),
        (vec![3, 3], vec![3, 3]),
    ];

    for (original_shape, expected_shape) in matrices {
        let size = original_shape[0] * original_shape[1];
        let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
        let x = Tensor::new(data, original_shape.clone());
        let xview = TensorView::new(&x);

        let y = xview.transpose().unwrap();
        let yview = TensorView::new(&y);

        assert_eq!(yview.shape, expected_shape);
    }
}

#[test]
fn transpose_identity_matrix() {
    let x = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], vec![3, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, xview.elements);
    assert_eq!(yview.shape, vec![3, 3]);
}

#[test]
fn transpose_zero_matrix() {
    let x = Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_negative_values() {
    let x = Tensor::new(vec![-1.0, -2.0, 3.0, 4.0, -5.0, 6.0], vec![2, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![-1.0, 4.0, -2.0, -5.0, 3.0, 6.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_decimal_precision() {
    let x = Tensor::new(vec![1.25, 2.75, 3.125, 4.875], vec![2, 2]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.25, 3.125, 2.75, 4.875]);
    assert_eq!(yview.shape, vec![2, 2]);
}

#[test]
fn transpose_single_row() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(yview.shape, vec![5, 1]);
}

#[test]
fn transpose_single_column() {
    let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![10.0, 20.0, 30.0]);
    assert_eq!(yview.shape, vec![1, 3]);
}

#[test]
fn transpose_repeated_elements() {
    let x = Tensor::new(vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0], vec![2, 3]);
    let xview = TensorView::new(&x);

    let y = xview.transpose().unwrap();
    let yview = TensorView::new(&y);

    assert_eq!(yview.elements, vec![5.0, 7.0, 5.0, 7.0, 5.0, 7.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}


#[test]
fn add_vector() {
    let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let xview = TensorView::new(&x);
    let yview = TensorView::new(&y);

    let result = xview.add(&yview).unwrap();

    assert_eq!(result.elements, vec![2.0, 2.0, 2.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn subtract_vector() {
    let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let xview = TensorView::new(&x);
    let yview = TensorView::new(&y);

    let result = xview.sub(&yview).unwrap();

    assert_eq!(result.elements, vec![0.0, -2.0, -4.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_vector() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let xview = TensorView::new(&x);

    let result = xview.mult(2.0).unwrap();

    assert_eq!(result.elements, vec![2.0, 4.0, 6.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn divide_vector() {
    let x = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]);
    let xview = TensorView::new(&x);

    let result = xview.div(2.0).unwrap();

    assert_eq!(result.elements, vec![1.0, 2.0, 3.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_matrix() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let xview = TensorView::new(&x);

    let result = xview.mult(3.0).unwrap();

    assert_eq!(result.elements, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    assert_eq!(result.shape, vec![2, 3]);
}

#[test]
fn divide_matrix() {
    let x = Tensor::new(vec![4.0, 8.0, 12.0, 16.0], vec![2, 2]);
    let xview = TensorView::new(&x);

    let result = xview.div(4.0).unwrap();

    assert_eq!(result.elements, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result.shape, vec![2, 2]);
}

#[test]
fn multiply_by_zero() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let xview = TensorView::new(&x);

    let result = xview.mult(0.0).unwrap();

    assert_eq!(result.elements, vec![0.0, 0.0, 0.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_by_negative() {
    let x = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]);
    let xview = TensorView::new(&x);

    let result = xview.mult(-2.0).unwrap();

    assert_eq!(result.elements, vec![-2.0, 4.0, -6.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn divide_by_decimal() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let xview = TensorView::new(&x);

    let result = xview.div(0.5).unwrap();

    assert_eq!(result.elements, vec![2.0, 4.0, 6.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_negative_values() {
    let x = Tensor::new(vec![-1.0, -2.0, -3.0, 4.0], vec![2, 2]);
    let xview = TensorView::new(&x);

    let result = xview.mult(3.0).unwrap();

    assert_eq!(result.elements, vec![-3.0, -6.0, -9.0, 12.0]);
    assert_eq!(result.shape, vec![2, 2]);
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

// Test cases for complex TensorView scenarios in add/subtract operations

#[test]
fn add_tensor_row_slices() {
    // Test adding row slices from the same tensor
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let row0 = view.row(0).unwrap(); // [1.0, 2.0, 3.0]
    let row1 = view.row(1).unwrap(); // [4.0, 5.0, 6.0]
    
    let result = row0.add(&row1).unwrap();
    
    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]
}

#[test]
fn subtract_tensor_row_slices() {
    // Test subtracting row slices from the same tensor
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let row0 = view.row(0).unwrap(); // [1.0, 2.0, 3.0]
    let row1 = view.row(1).unwrap(); // [4.0, 5.0, 6.0]
    
    let result = row1.sub(&row0).unwrap();
    
    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![3.0, 3.0, 3.0]); // [4-1, 5-2, 6-3]
}

#[test]
fn add_tensor_column_slices() {
    // Test adding column slices (non-contiguous views) - this will expose stride bugs
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let col0 = view.column(0).unwrap(); // [1.0, 4.0] (non-contiguous)
    let col1 = view.column(1).unwrap(); // [2.0, 5.0] (non-contiguous)
    
    let result = col0.add(&col1).unwrap();
    
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![3.0, 9.0]); // [1+2, 4+5]
}

#[test]
fn subtract_tensor_column_slices() {
    // Test subtracting column slices (non-contiguous views)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let col0 = view.column(0).unwrap(); // [1.0, 4.0]
    let col2 = view.column(2).unwrap(); // [3.0, 6.0]
    
    let result = col2.sub(&col0).unwrap();
    
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![2.0, 2.0]); // [3-1, 6-4]
}

#[test]
fn add_shape_mismatch_should_error() {
    // Test that shape mismatches properly return errors
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![1.0, 2.0], vec![2]);
    
    let view_a = TensorView::new(&tensor_a);
    let view_b = TensorView::new(&tensor_b);
    
    let result = view_a.add(&view_b);
    assert!(result.is_err()); // Should fail due to shape mismatch
}

#[test]
fn subtract_shape_mismatch_should_error() {
    // Test that shape mismatches properly return errors
    let tensor_2d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let tensor_1d = Tensor::new(vec![1.0, 2.0], vec![2]);
    
    let view_2d = TensorView::new(&tensor_2d);
    let view_1d = TensorView::new(&tensor_1d);
    
    let result = view_2d.sub(&view_1d);
    assert!(result.is_err()); // Should fail due to rank mismatch
}

#[test]
fn add_row_column_mismatch_should_error() {
    // Test mixing row and column operations with different shapes
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let row = view.row(0).unwrap();    // shape: [3]
    let col = view.column(0).unwrap(); // shape: [2]
    
    let result = row.add(&col);
    assert!(result.is_err()); // Should fail due to shape mismatch [3] vs [2]
}

#[test]
fn add_views_from_different_tensors() {
    // Test operations between views from different underlying tensors
    // This exposes the bug where we assume same underlying storage size
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]); // Different size storage
    
    let view_a = TensorView::new(&tensor_a);
    let view_b_slice = tensor_b.elements.get(0..3).unwrap(); // Create view of first 3 elements
    
    let view_b = TensorView {
        elements: view_b_slice,
        shape: vec![3],
        strides: vec![1],
    };
    
    let result = view_a.add(&view_b).unwrap();
    
    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![11.0, 22.0, 33.0]); // [1+10, 2+20, 3+30]
}

#[test]
fn subtract_views_different_underlying_storage() {
    // Another test with different underlying storage sizes and different data
    let small_tensor = Tensor::new(vec![5.0, 15.0], vec![2]);
    let large_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);
    
    let small_view = TensorView::new(&small_tensor);
    let large_row = large_tensor.elements.get(2..4).unwrap(); // Elements [3.0, 4.0]
    
    let large_view = TensorView {
        elements: large_row,
        shape: vec![2],
        strides: vec![1],
    };
    
    let result = small_view.sub(&large_view).unwrap();
    
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![2.0, 11.0]); // [5-3, 15-4]
}

#[test]
fn add_3d_tensor_slices() {
    // Test adding slices from 3D tensors - complex stride patterns
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
        vec![2, 2, 2]
    );
    let view = TensorView::new(&tensor);
    
    let slice0 = view.at(0).unwrap(); // First 2x2 slice: [[1,2], [3,4]]
    let slice1 = view.at(1).unwrap(); // Second 2x2 slice: [[5,6], [7,8]]
    
    let result = slice0.add(&slice1).unwrap();
    
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.elements, vec![6.0, 8.0, 10.0, 12.0]); // [1+5, 2+6, 3+7, 4+8]
}

#[test]
fn subtract_3d_tensor_slices() {
    // Test subtracting slices from 3D tensors
    let tensor = Tensor::new(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 
        vec![2, 2, 2]
    );
    let view = TensorView::new(&tensor);
    
    let slice1 = view.at(1).unwrap(); // Second slice: [[50,60], [70,80]]
    let slice0 = view.at(0).unwrap(); // First slice: [[10,20], [30,40]]
    
    let result = slice1.sub(&slice0).unwrap();
    
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.elements, vec![40.0, 40.0, 40.0, 40.0]); // [50-10, 60-20, 70-30, 80-40]
}

#[test]
fn add_full_view_with_row_slice() {
    // Test mixing full tensor view with a row slice - different view types
    let tensor_full = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let tensor_2d = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    
    let full_view = TensorView::new(&tensor_full);
    let view_2d = TensorView::new(&tensor_2d);
    let row_view = view_2d.row(1).unwrap(); // [40.0, 50.0, 60.0]
    
    let result = full_view.add(&row_view).unwrap();
    
    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![41.0, 51.0, 61.0]); // [1+40, 1+50, 1+60]
}

#[test]
fn subtract_column_from_full_view() {
    // Test subtracting a column slice from a full vector view
    let tensor_vec = Tensor::new(vec![100.0, 200.0], vec![2]);
    let tensor_matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    
    let vec_view = TensorView::new(&tensor_vec);
    let matrix_view = TensorView::new(&tensor_matrix);
    let col_view = matrix_view.column(1).unwrap(); // [2.0, 5.0]
    
    let result = vec_view.sub(&col_view).unwrap();
    
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![98.0, 195.0]); // [100-2, 200-5]
}

#[test]
fn add_nested_slices() {
    // Test adding slices that are themselves derived from sliced views
    let tensor_3d = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3, 2]
    );
    let view_3d = TensorView::new(&tensor_3d);
    
    let slice0 = view_3d.at(0).unwrap(); // First 3x2 slice
    let slice1 = view_3d.at(1).unwrap(); // Second 3x2 slice
    
    let row_from_slice0 = slice0.at(1).unwrap(); // Second row from first slice: [3.0, 4.0]
    let row_from_slice1 = slice1.at(0).unwrap(); // First row from second slice: [7.0, 8.0]
    
    let result = row_from_slice0.add(&row_from_slice1).unwrap();
    
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![10.0, 12.0]); // [3+7, 4+8]
}

#[test]
fn sum_vector() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 10.0); // 1 + 2 + 3 + 4 = 10
}

#[test]
fn sum_matrix() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 21.0); // 1 + 2 + 3 + 4 + 5 + 6 = 21
}

#[test]
fn sum_single_element() {
    let tensor = Tensor::new(vec![42.0], vec![1]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 42.0);
}

#[test]
fn sum_negative_values() {
    let tensor = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 2.0); // -1 + 2 + (-3) + 4 = 2
}

#[test]
fn mean_vector() {
    let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
    let view = TensorView::new(&tensor);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 5.0); // (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
}

#[test]
fn mean_matrix() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 3.5); // (1 + 2 + 3 + 4 + 5 + 6) / 6 = 21 / 6 = 3.5
}

#[test]
fn mean_single_element() {
    let tensor = Tensor::new(vec![7.5], vec![1]);
    let view = TensorView::new(&tensor);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 7.5);
}

#[test]
fn mean_with_decimals() {
    let tensor = Tensor::new(vec![1.5, 2.5, 3.0], vec![3]);
    let view = TensorView::new(&tensor);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 7.0 / 3.0); // (1.5 + 2.5 + 3.0) / 3 = 7.0 / 3
}

#[test]
fn sum_row_slice() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let row0 = view.row(0).unwrap(); // [1.0, 2.0, 3.0]
    let result = row0.sum().unwrap();
    assert_eq!(result, 6.0); // 1 + 2 + 3 = 6
    
    let row1 = view.row(1).unwrap(); // [4.0, 5.0, 6.0]
    let result = row1.sum().unwrap();
    assert_eq!(result, 15.0); // 4 + 5 + 6 = 15
}

#[test]
fn mean_row_slice() {
    let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let row0 = view.row(0).unwrap(); // [2.0, 4.0, 6.0]
    let result = row0.mean().unwrap();
    assert_eq!(result, 4.0); // (2 + 4 + 6) / 3 = 12 / 3 = 4
    
    let row1 = view.row(1).unwrap(); // [8.0, 10.0, 12.0]
    let result = row1.mean().unwrap();
    assert_eq!(result, 10.0); // (8 + 10 + 12) / 3 = 30 / 3 = 10
}

#[test]
fn sum_row_slice_different_shapes() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let view = TensorView::new(&tensor);
    
    let row0 = view.row(0).unwrap(); // [1.0, 2.0, 3.0, 4.0]
    let result = row0.sum().unwrap();
    assert_eq!(result, 10.0); // 1 + 2 + 3 + 4 = 10
    
    let row1 = view.row(1).unwrap(); // [5.0, 6.0, 7.0, 8.0]
    let result = row1.sum().unwrap();
    assert_eq!(result, 26.0); // 5 + 6 + 7 + 8 = 26
}

#[test]
fn sum_column_slice() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let col0 = view.column(0).unwrap(); // [1.0, 4.0] (non-contiguous)
    let result = col0.sum().unwrap();
    assert_eq!(result, 5.0); // 1 + 4 = 5
    
    let col1 = view.column(1).unwrap(); // [2.0, 5.0] (non-contiguous)
    let result = col1.sum().unwrap();
    assert_eq!(result, 7.0); // 2 + 5 = 7
    
    let col2 = view.column(2).unwrap(); // [3.0, 6.0] (non-contiguous)
    let result = col2.sum().unwrap();
    assert_eq!(result, 9.0); // 3 + 6 = 9
}

#[test]
fn mean_column_slice() {
    let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3]);
    let view = TensorView::new(&tensor);
    
    let col0 = view.column(0).unwrap(); // [2.0, 8.0] (non-contiguous)
    let result = col0.mean().unwrap();
    assert_eq!(result, 5.0); // (2 + 8) / 2 = 10 / 2 = 5
    
    let col2 = view.column(2).unwrap(); // [6.0, 12.0] (non-contiguous)
    let result = col2.mean().unwrap();
    assert_eq!(result, 9.0); // (6 + 12) / 2 = 18 / 2 = 9
}

#[test]
fn sum_column_slice_larger_matrix() {
    let tensor = Tensor::new(vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ], vec![3, 4]);
    let view = TensorView::new(&tensor);
    
    let col1 = view.column(1).unwrap(); // [2.0, 6.0, 10.0] (stride = 4)
    let result = col1.sum().unwrap();
    assert_eq!(result, 18.0); // 2 + 6 + 10 = 18
    
    let col3 = view.column(3).unwrap(); // [4.0, 8.0, 12.0] (stride = 4)
    let result = col3.sum().unwrap();
    assert_eq!(result, 24.0); // 4 + 8 + 12 = 24
}

#[test]
fn sum_3d_tensor_slice() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2]
    );
    let view = TensorView::new(&tensor);
    
    let slice0 = view.at(0).unwrap(); // First 2x2 slice: [[1,2], [3,4]]
    let result = slice0.sum().unwrap();
    assert_eq!(result, 10.0); // 1 + 2 + 3 + 4 = 10
    
    let slice1 = view.at(1).unwrap(); // Second 2x2 slice: [[5,6], [7,8]]
    let result = slice1.sum().unwrap();
    assert_eq!(result, 26.0); // 5 + 6 + 7 + 8 = 26
}

#[test]
fn mean_3d_tensor_slice() {
    let tensor = Tensor::new(
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        vec![2, 2, 2]
    );
    let view = TensorView::new(&tensor);
    
    let slice0 = view.at(0).unwrap(); // First slice: [[2,4], [6,8]]
    let result = slice0.mean().unwrap();
    assert_eq!(result, 5.0); // (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
    
    let slice1 = view.at(1).unwrap(); // Second slice: [[10,12], [14,16]]
    let result = slice1.mean().unwrap();
    assert_eq!(result, 13.0); // (10 + 12 + 14 + 16) / 4 = 52 / 4 = 13
}

#[test]
fn sum_nested_3d_slices() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3, 2]
    );
    let view = TensorView::new(&tensor);
    
    let slice0 = view.at(0).unwrap(); // First 3x2 slice
    let row_from_slice0 = slice0.at(1).unwrap(); // Second row: [3.0, 4.0]
    let result = row_from_slice0.sum().unwrap();
    assert_eq!(result, 7.0); // 3 + 4 = 7
    
    let slice1 = view.at(1).unwrap(); // Second 3x2 slice  
    let row_from_slice1 = slice1.at(2).unwrap(); // Third row: [11.0, 12.0]
    let result = row_from_slice1.sum().unwrap();
    assert_eq!(result, 23.0); // 11 + 12 = 23
}

#[test]
fn sum_zeros() {
    let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 0.0);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn sum_mean_mixed_signs() {
    let tensor = Tensor::new(vec![-5.0, 10.0, -3.0, 8.0], vec![4]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 10.0); // -5 + 10 + (-3) + 8 = 10
    
    let result = view.mean().unwrap();
    assert_eq!(result, 2.5); // 10 / 4 = 2.5
}

#[test]
fn sum_mean_large_values() {
    let tensor = Tensor::new(vec![1000.0, 2000.0, 3000.0], vec![3]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert_eq!(result, 6000.0);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 2000.0); // 6000 / 3 = 2000
}

#[test]
fn sum_mean_fractional_values() {
    let tensor = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    let view = TensorView::new(&tensor);
    
    let result = view.sum().unwrap();
    assert!((result - 1.0).abs() < 1e-6); // 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    
    let result = view.mean().unwrap();
    assert!((result - 0.25).abs() < 1e-6); // 1.0 / 4 = 0.25
}
