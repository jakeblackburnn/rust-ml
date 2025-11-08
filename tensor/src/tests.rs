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
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_2x2_square() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(yview.shape, vec![2, 2]);
}

#[test]
fn transpose_3x3_square() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    assert_eq!(yview.shape, vec![3, 3]);
}

#[test]
fn transpose_1x4_rectangular() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(yview.shape, vec![4, 1]);
}

#[test]
fn transpose_4x1_rectangular() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(yview.shape, vec![1, 4]);
}

#[test]
fn transpose_2x5_rectangular() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 5]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]);
    assert_eq!(yview.shape, vec![5, 2]);
}

#[test]
fn transpose_1x1_single_element() {
    let x = Tensor::new(vec![42.0], vec![1, 1]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![42.0]);
    assert_eq!(yview.shape, vec![1, 1]);
}

#[test]
fn transpose_double_transpose_identity() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();
    let z = yview.transpose().unwrap();
    let zview = z.view();

    assert_eq!(zview.elements, xview.elements);
    assert_eq!(zview.shape, xview.shape);
}

#[test]
fn transpose_element_position_mapping() {
    let x = Tensor::new(vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0], vec![3, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

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
        let xview = x.view();

        let y = xview.transpose().unwrap();
        let yview = y.view();

        assert_eq!(yview.shape, expected_shape);
    }
}

#[test]
fn transpose_identity_matrix() {
    let x = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], vec![3, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, xview.elements);
    assert_eq!(yview.shape, vec![3, 3]);
}

#[test]
fn transpose_zero_matrix() {
    let x = Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_negative_values() {
    let x = Tensor::new(vec![-1.0, -2.0, 3.0, 4.0, -5.0, 6.0], vec![2, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![-1.0, 4.0, -2.0, -5.0, 3.0, 6.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_decimal_precision() {
    let x = Tensor::new(vec![1.25, 2.75, 3.125, 4.875], vec![2, 2]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.25, 3.125, 2.75, 4.875]);
    assert_eq!(yview.shape, vec![2, 2]);
}

#[test]
fn transpose_single_row() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(yview.shape, vec![5, 1]);
}

#[test]
fn transpose_single_column() {
    let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![10.0, 20.0, 30.0]);
    assert_eq!(yview.shape, vec![1, 3]);
}

#[test]
fn transpose_repeated_elements() {
    let x = Tensor::new(vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0], vec![2, 3]);
    let xview = x.view();

    let y = xview.transpose().unwrap();
    let yview = y.view();

    assert_eq!(yview.elements, vec![5.0, 7.0, 5.0, 7.0, 5.0, 7.0]);
    assert_eq!(yview.shape, vec![3, 2]);
}

#[test]
fn transpose_not_matrix() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let result = view.transpose();
    assert!(result.is_err());
    
    if let Err(TensorError::NotMatrixError(rank)) = result {
        assert_eq!(rank, 1);
        
        let error_message = format!("{}", TensorError::NotMatrixError(rank));
        assert_eq!(error_message, "Expected Matrix, found rank 1 tensor.");
    } else {
        panic!("Expected NotMatrixError");
    }
}

#[test]
fn add_vector() {
    let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.add(&yview).unwrap();

    assert_eq!(result.elements, vec![2.0, 2.0, 2.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn subtract_vector() {
    let x = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.sub(&yview).unwrap();

    assert_eq!(result.elements, vec![0.0, -2.0, -4.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn add_shape_mismatch() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![1.0, 2.0], vec![2]);
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.add(&view_b);
    assert!(result.is_err());
    
    if let Err(TensorError::ShapeMismatch { provided, expected }) = result {
        assert_eq!(provided, vec![2]);
        assert_eq!(expected, vec![3]);
        
        let error_message = format!("{}", TensorError::ShapeMismatch { provided, expected });
        assert_eq!(error_message, "Tensor shapes did not match. found (other.shape): [2], expected (self.shape): [3]");
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

#[test]
fn sub_shape_mismatch() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let tensor_b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.sub(&view_b);
    assert!(result.is_err());
    
    if let Err(TensorError::ShapeMismatch { provided, expected }) = result {
        assert_eq!(provided, vec![3]);
        assert_eq!(expected, vec![2, 2]);
        
        let error_message = format!("{}", TensorError::ShapeMismatch { provided, expected });
        assert_eq!(error_message, "Tensor shapes did not match. found (other.shape): [3], expected (self.shape): [2, 2]");
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

#[test]
fn multiply_vector() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).view().mult(2.0).unwrap();

    assert_eq!(result.elements, vec![2.0, 4.0, 6.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn divide_vector() {
    let result = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]).view().div(2.0).unwrap();

    assert_eq!(result.elements, vec![1.0, 2.0, 3.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_matrix() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).view().mult(3.0).unwrap();

    assert_eq!(result.elements, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    assert_eq!(result.shape, vec![2, 3]);
}

#[test]
fn divide_matrix() {
    let result = Tensor::new(vec![4.0, 8.0, 12.0, 16.0], vec![2, 2]).view().div(4.0).unwrap();

    assert_eq!(result.elements, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result.shape, vec![2, 2]);
}

#[test]
fn multiply_by_zero() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).view().mult(0.0).unwrap();

    assert_eq!(result.elements, vec![0.0, 0.0, 0.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_by_negative() {
    let result = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]).view().mult(-2.0).unwrap();

    assert_eq!(result.elements, vec![-2.0, 4.0, -6.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn divide_by_decimal() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).view().div(0.5).unwrap();

    assert_eq!(result.elements, vec![2.0, 4.0, 6.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn multiply_negative_values() {
    let result = Tensor::new(vec![-1.0, -2.0, -3.0, 4.0], vec![2, 2]).view().mult(3.0).unwrap();

    assert_eq!(result.elements, vec![-3.0, -6.0, -9.0, 12.0]);
    assert_eq!(result.shape, vec![2, 2]);
}

#[test]
fn tensor_view_constructor() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();

    assert_eq!(view.shape, vec![2, 3]);
    assert_eq!(view.strides, vec![3, 1]);
    assert_eq!(view.elements.len(), 6);
    assert_eq!(view.elements, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tensor_view_at_2d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

    assert_eq!(view.get(&[0]).unwrap(), 1.0);
    assert_eq!(view.get(&[1]).unwrap(), 2.0);
    assert_eq!(view.get(&[2]).unwrap(), 3.0);
}

#[test]
fn tensor_view_get_2d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

    assert_eq!(view.get(&[0]).unwrap(), 42.0);

    let result = view.get(&[1]);
    assert!(result.is_err());
}

#[test]
fn tensor_view_dot_product() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();

    let result = view_a.dot(&view_b).unwrap();
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

#[test]
fn tensor_view_dot_product_rank_mismatch() {
    let tensor_1d = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_2d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let view_1d = tensor_1d.view();
    let view_2d = tensor_2d.view();

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
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();

    let result = view_a.dot(&view_b);
    assert!(result.is_err());
    
    // Should return RankMismatch error (used as placeholder for ShapeMismatch)
    if let Err(TensorError::RankMismatch { provided, expected }) = result {
        assert_eq!(provided, 2);
        assert_eq!(expected, 3);
    }
}

#[test]
fn dot_shape_mismatch() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![1.0, 2.0], vec![2]);
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.dot(&view_b);
    assert!(result.is_err());
    
    if let Err(TensorError::ShapeMismatch { provided, expected }) = result {
        assert_eq!(provided, vec![2]);
        assert_eq!(expected, vec![3]);
        
        let error_message = format!("{}", TensorError::ShapeMismatch { provided, expected });
        assert_eq!(error_message, "Tensor shapes did not match. found (other.shape): [2], expected (self.shape): [3]");
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

#[test]
fn dot_not_vector() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let tensor_b = Tensor::new(vec![1.0, 2.0], vec![2]);
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.dot(&view_b);
    assert!(result.is_err());
    
    if let Err(TensorError::NotVectorError(rank)) = result {
        assert_eq!(rank, 2);
        
        let error_message = format!("{}", TensorError::NotVectorError(rank));
        assert_eq!(error_message, "Expected Vector, found rank 2 tensor.");
    } else {
        panic!("Expected NotVectorError");
    }
}

#[test]
fn tensor_view_column_extraction() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    let view = tensor.view();

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
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();

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

#[test]
fn matmul_not_matrix() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.matmul(&view_b);
    assert!(result.is_err());
    
    if let Err(TensorError::NotMatrixError(rank)) = result {
        assert_eq!(rank, 1);
        
        let error_message = format!("{}", TensorError::NotMatrixError(rank));
        assert_eq!(error_message, "Expected Matrix, found rank 1 tensor.");
    } else {
        panic!("Expected NotMatrixError");
    }
}

#[test]
fn matmul_incompatible_dimensions() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let tensor_b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.matmul(&view_b);
    assert!(result.is_err());
    
    if let Err(TensorError::IncompatibleDimensions { k1, k2 }) = result {
        assert_eq!(k1, 3);
        assert_eq!(k2, 2);
        
        let error_message = format!("{}", TensorError::IncompatibleDimensions { k1, k2 });
        assert_eq!(error_message, "Incompatible inner dimentions: k1 (self.shape[1]): 3, k2 (other.shape[0]): 2");
    } else {
        panic!("Expected IncompatibleDimensions error");
    }
}

// Test cases for complex TensorView scenarios in add/subtract operations

#[test]
fn add_tensor_row_slices() {
    // Test adding row slices from the same tensor
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    
    let view_a = tensor_a.view();
    let view_b = tensor_b.view();
    
    let result = view_a.add(&view_b);
    assert!(result.is_err()); // Should fail due to shape mismatch
}

#[test]
fn subtract_shape_mismatch_should_error() {
    // Test that shape mismatches properly return errors
    let tensor_2d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let tensor_1d = Tensor::new(vec![1.0, 2.0], vec![2]);
    
    let view_2d = tensor_2d.view();
    let view_1d = tensor_1d.view();
    
    let result = view_2d.sub(&view_1d);
    assert!(result.is_err()); // Should fail due to rank mismatch
}

#[test]
fn add_row_column_mismatch_should_error() {
    // Test mixing row and column operations with different shapes
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
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
    
    let view_a = tensor_a.view();
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
    
    let small_view = small_tensor.view();
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
fn elem_wise_mult_vector() {
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let y = Tensor::new(vec![5.0, 6.0, 7.0], vec![3]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.elem_wise_mult(&yview).unwrap();

    assert_eq!(result.elements, vec![10.0, 18.0, 28.0]); // [2*5, 3*6, 4*7]
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn elem_wise_mult_matrix() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.elem_wise_mult(&yview).unwrap();

    assert_eq!(result.elements, vec![5.0, 12.0, 21.0, 32.0]); // [1*5, 2*6, 3*7, 4*8]
    assert_eq!(result.shape, vec![2, 2]);
}

#[test]
fn elem_wise_mult_with_zeros() {
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let y = Tensor::new(vec![0.0, 6.0, 0.0], vec![3]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.elem_wise_mult(&yview).unwrap();

    assert_eq!(result.elements, vec![0.0, 18.0, 0.0]); // [2*0, 3*6, 4*0]
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn elem_wise_mult_with_negatives() {
    let x = Tensor::new(vec![2.0, -3.0, 4.0], vec![3]);
    let y = Tensor::new(vec![-5.0, 6.0, -7.0], vec![3]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.elem_wise_mult(&yview).unwrap();

    assert_eq!(result.elements, vec![-10.0, -18.0, -28.0]); // [2*-5, -3*6, 4*-7]
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn elem_wise_mult_identity() {
    let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let y = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);

    let xview = x.view();
    let yview = y.view();

    let result = xview.elem_wise_mult(&yview).unwrap();

    assert_eq!(result.elements, vec![2.0, 3.0, 4.0]); // [2*1, 3*1, 4*1]
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn elem_wise_mult_shape_mismatch() {
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let tensor_b = Tensor::new(vec![1.0, 2.0], vec![2]);

    let view_a = tensor_a.view();
    let view_b = tensor_b.view();

    let result = view_a.elem_wise_mult(&view_b);
    assert!(result.is_err());

    if let Err(TensorError::ShapeMismatch { provided, expected }) = result {
        assert_eq!(provided, vec![2]);
        assert_eq!(expected, vec![3]);
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

#[test]
fn elem_wise_mult_row_slices() {
    // Test multiplying row slices from the same tensor
    let tensor = Tensor::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3]);
    let view = tensor.view();

    let row0 = view.row(0).unwrap(); // [2.0, 3.0, 4.0]
    let row1 = view.row(1).unwrap(); // [5.0, 6.0, 7.0]

    let result = row0.elem_wise_mult(&row1).unwrap();

    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![10.0, 18.0, 28.0]); // [2*5, 3*6, 4*7]
}

#[test]
fn elem_wise_mult_column_slices() {
    // Test multiplying column slices (non-contiguous views)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();

    let col0 = view.column(0).unwrap(); // [1.0, 4.0] (non-contiguous)
    let col1 = view.column(1).unwrap(); // [2.0, 5.0] (non-contiguous)

    let result = col0.elem_wise_mult(&col1).unwrap();

    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![2.0, 20.0]); // [1*2, 4*5]
}

#[test]
fn add_3d_tensor_slices() {
    // Test adding slices from 3D tensors - complex stride patterns
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
        vec![2, 2, 2]
    );
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    
    let full_view = tensor_full.view();
    let view_2d = tensor_2d.view();
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
    
    let vec_view = tensor_vec.view();
    let matrix_view = tensor_matrix.view();
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
    let view_3d = tensor_3d.view();
    
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
    let result = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).view().sum().unwrap();
    assert_eq!(result, 10.0); // 1 + 2 + 3 + 4 = 10
}

#[test]
fn sum_matrix() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).view().sum().unwrap();
    assert_eq!(result, 21.0); // 1 + 2 + 3 + 4 + 5 + 6 = 21
}

#[test]
fn sum_single_element() {
    let result = Tensor::new(vec![42.0], vec![1]).view().sum().unwrap();
    assert_eq!(result, 42.0);
}

#[test]
fn sum_negative_values() {
    let result = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]).view().sum().unwrap();
    assert_eq!(result, 2.0); // -1 + 2 + (-3) + 4 = 2
}

#[test]
fn mean_vector() {
    let result = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4]).view().mean().unwrap();
    assert_eq!(result, 5.0); // (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
}

#[test]
fn mean_matrix() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).view().mean().unwrap();
    assert_eq!(result, 3.5); // (1 + 2 + 3 + 4 + 5 + 6) / 6 = 21 / 6 = 3.5
}

#[test]
fn mean_single_element() {
    let result = Tensor::new(vec![7.5], vec![1]).view().mean().unwrap();
    assert_eq!(result, 7.5);
}

#[test]
fn mean_with_decimals() {
    let result = Tensor::new(vec![1.5, 2.5, 3.0], vec![3]).view().mean().unwrap();
    assert_eq!(result, 7.0 / 3.0); // (1.5 + 2.5 + 3.0) / 3 = 7.0 / 3
}

#[test]
fn sum_row_slice() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
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
    let view = tensor.view();
    
    let result = view.sum().unwrap();
    assert_eq!(result, 0.0);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn sum_mean_mixed_signs() {
    let tensor = Tensor::new(vec![-5.0, 10.0, -3.0, 8.0], vec![4]);
    let view = tensor.view();
    
    let result = view.sum().unwrap();
    assert_eq!(result, 10.0); // -5 + 10 + (-3) + 8 = 10
    
    let result = view.mean().unwrap();
    assert_eq!(result, 2.5); // 10 / 4 = 2.5
}

#[test]
fn sum_mean_large_values() {
    let tensor = Tensor::new(vec![1000.0, 2000.0, 3000.0], vec![3]);
    let view = tensor.view();
    
    let result = view.sum().unwrap();
    assert_eq!(result, 6000.0);
    
    let result = view.mean().unwrap();
    assert_eq!(result, 2000.0); // 6000 / 3 = 2000
}

#[test]
fn sum_mean_fractional_values() {
    let tensor = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    let view = tensor.view();
    
    let result = view.sum().unwrap();
    assert!((result - 1.0).abs() < 1e-6); // 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    
    let result = view.mean().unwrap();
    assert!((result - 0.25).abs() < 1e-6); // 1.0 / 4 = 0.25
}

#[test]
fn sum_axis_matrix_axis_0() {
    // Sum along axis 0 (sum down rows)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    let result = view.sum_axis(0).unwrap();

    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]
}

#[test]
fn sum_axis_matrix_axis_1() {
    // Sum along axis 1 (sum across columns)
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    let result = view.sum_axis(1).unwrap();

    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
}

#[test]
fn sum_axis_3d_axis_0() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2]
    );
    let view = tensor.view();
    let result = view.sum_axis(0).unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.elements, vec![6.0, 8.0, 10.0, 12.0]); // [1+5, 2+6, 3+7, 4+8]
}

#[test]
fn sum_axis_3d_axis_1() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2]
    );
    let view = tensor.view();
    let result = view.sum_axis(1).unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.elements, vec![4.0, 6.0, 12.0, 14.0]); // [1+3, 2+4, 5+7, 6+8]
}

#[test]
fn sum_axis_3d_axis_2() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2]
    );
    let view = tensor.view();
    let result = view.sum_axis(2).unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.elements, vec![3.0, 7.0, 11.0, 15.0]); // [1+2, 3+4, 5+6, 7+8]
}

#[test]
fn sum_axis_invalid_axis() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let view = tensor.view();

    let result = view.sum_axis(2);
    assert!(result.is_err());

    if let Err(TensorError::CoordsOutOfBounds { rank: _, provided, max }) = result {
        assert_eq!(provided, 2);
        assert_eq!(max, 1);
    } else {
        panic!("Expected CoordsOutOfBounds error");
    }
}

#[test]
fn sum_axis_vector() {
    // Summing a vector along axis 0 should give a scalar
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let view = tensor.view();
    let result = view.sum_axis(0).unwrap();

    assert_eq!(result.shape, vec![1]);
    assert_eq!(result.elements, vec![10.0]); // 1+2+3+4
}

#[test]
fn sum_axis_with_zeros() {
    let tensor = Tensor::new(vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0], vec![2, 3]);
    let view = tensor.view();
    let result = view.sum_axis(0).unwrap();

    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.elements, vec![1.0, 5.0, 3.0]); // [1+0, 0+5, 3+0]
}

#[test]
fn sum_axis_with_negatives() {
    let tensor = Tensor::new(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], vec![2, 3]);
    let view = tensor.view();
    let result = view.sum_axis(1).unwrap();

    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.elements, vec![2.0, -5.0]); // [1-2+3, -4+5-6]
}

#[test]
fn sum_axis_row_slice() {
    // Test summing a row slice (should reduce to scalar with shape [1])
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    let row = view.row(0).unwrap(); // [1.0, 2.0, 3.0]

    let result = row.sum_axis(0).unwrap();
    assert_eq!(result.shape, vec![1]);
    assert_eq!(result.elements, vec![6.0]); // 1+2+3
}

#[test]
fn sum_axis_total_equals_sum() {
    // Verify that sum_axis gives same total as sum() method
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();

    let total_sum = view.sum().unwrap();

    // Sum along axis 0, then sum the result
    let partial_sum = view.sum_axis(0).unwrap();
    let final_sum = partial_sum.view().sum().unwrap();

    assert_eq!(total_sum, final_sum);
    assert_eq!(final_sum, 21.0);
}

#[test]
fn zeroes_1d() {
    let tensor = Tensor::zeroes(vec![3]);
    assert_eq!(tensor.shape, vec![3]);
    assert_eq!(tensor.elements, vec![0.0, 0.0, 0.0]);
}

#[test]
fn zeroes_2d() {
    let tensor = Tensor::zeroes(vec![2, 3]);
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.elements, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeroes_3d() {
    let tensor = Tensor::zeroes(vec![2, 2, 2]);
    assert_eq!(tensor.shape, vec![2, 2, 2]);
    assert_eq!(tensor.elements, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeroes_single_element() {
    let tensor = Tensor::zeroes(vec![1]);
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.elements, vec![0.0]);
}

#[test]
fn zeroes_large_tensor() {
    let tensor = Tensor::zeroes(vec![5, 4]);
    assert_eq!(tensor.shape, vec![5, 4]);
    assert_eq!(tensor.elements.len(), 20);
    assert!(tensor.elements.iter().all(|&x| x == 0.0));
}

#[test]
fn ones_1d() {
    let tensor = Tensor::ones(vec![3]);
    assert_eq!(tensor.shape, vec![3]);
    assert_eq!(tensor.elements, vec![1.0, 1.0, 1.0]);
}

#[test]
fn ones_2d() {
    let tensor = Tensor::ones(vec![2, 3]);
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.elements, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn ones_3d() {
    let tensor = Tensor::ones(vec![2, 2, 2]);
    assert_eq!(tensor.shape, vec![2, 2, 2]);
    assert_eq!(tensor.elements, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn ones_single_element() {
    let tensor = Tensor::ones(vec![1]);
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.elements, vec![1.0]);
}

#[test]
fn ones_large_tensor() {
    let tensor = Tensor::ones(vec![5, 4]);
    assert_eq!(tensor.shape, vec![5, 4]);
    assert_eq!(tensor.elements.len(), 20);
    assert!(tensor.elements.iter().all(|&x| x == 1.0));
}

#[test]
fn slice_1d_vector_basic() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let view = tensor.view();
    
    let slice = view.slice(1, 4).unwrap();
    assert_eq!(slice.shape, vec![3]);
    assert_eq!(slice.elements, &[2.0, 3.0, 4.0]);
}

#[test]
fn slice_1d_vector_start() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let view = tensor.view();
    
    let slice = view.slice(0, 2).unwrap();
    assert_eq!(slice.shape, vec![2]);
    assert_eq!(slice.elements, &[1.0, 2.0]);
}

#[test]
fn slice_1d_vector_end() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let view = tensor.view();
    
    let slice = view.slice(2, 4).unwrap();
    assert_eq!(slice.shape, vec![2]);
    assert_eq!(slice.elements, &[3.0, 4.0]);
}

#[test]
fn slice_1d_single_element() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let slice = view.slice(1, 2).unwrap();
    assert_eq!(slice.shape, vec![1]);
    assert_eq!(slice.elements, &[2.0]);
}

#[test]
fn slice_1d_full_range() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let slice = view.slice(0, 3).unwrap();
    assert_eq!(slice.shape, vec![3]);
    assert_eq!(slice.elements, &[1.0, 2.0, 3.0]);
}

#[test]
fn slice_2d_matrix_columns() {
    // Matrix: [[1, 2, 3, 4], [5, 6, 7, 8]]
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let view = tensor.view();
    
    let slice = view.slice(1, 3).unwrap();
    assert_eq!(slice.shape, vec![2, 2]);
    
    // Should get columns 1-2: [[2, 3], [6, 7]]
    assert_eq!(slice.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(slice.get(&[0, 1]).unwrap(), 3.0);
    assert_eq!(slice.get(&[1, 0]).unwrap(), 6.0);
    assert_eq!(slice.get(&[1, 1]).unwrap(), 7.0);
}

#[test]
fn slice_2d_first_columns() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
    let slice = view.slice(0, 2).unwrap();
    assert_eq!(slice.shape, vec![2, 2]);
    
    // Should get first 2 columns: [[1, 2], [4, 5]]
    assert_eq!(slice.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(slice.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(slice.get(&[1, 0]).unwrap(), 4.0);
    assert_eq!(slice.get(&[1, 1]).unwrap(), 5.0);
}

#[test]
fn slice_2d_last_columns() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
    let slice = view.slice(1, 3).unwrap();
    assert_eq!(slice.shape, vec![2, 2]);
    
    // Should get columns 1-2: [[2, 3], [5, 6]]
    assert_eq!(slice.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(slice.get(&[0, 1]).unwrap(), 3.0);
    assert_eq!(slice.get(&[1, 0]).unwrap(), 5.0);
    assert_eq!(slice.get(&[1, 1]).unwrap(), 6.0);
}

#[test]
fn slice_2d_single_column() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
    let slice = view.slice(1, 2).unwrap();
    assert_eq!(slice.shape, vec![2, 1]);
    
    // Should get middle column: [[2], [5]]
    assert_eq!(slice.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(slice.get(&[1, 0]).unwrap(), 5.0);
}

#[test]
fn slice_row_then_slice() {
    // Test slicing a row slice - should work for feature extraction
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let view = tensor.view();
    
    let row = view.row(0).unwrap(); // [1, 2, 3, 4]
    let features = row.slice(0, 3).unwrap(); // First 3 elements as features
    
    assert_eq!(features.shape, vec![3]);
    assert_eq!(features.elements, &[1.0, 2.0, 3.0]);
    
    let target = row.get(&[3]).unwrap(); // Last element as target
    assert_eq!(target, 4.0);
}

#[test]
fn slice_out_of_bounds_start() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let result = view.slice(3, 4);
    assert!(result.is_err());
    
    if let Err(TensorError::CoordsOutOfBounds { rank, provided, max }) = result {
        assert_eq!(rank, 1);
        assert_eq!(provided, 3);
        assert_eq!(max, 2);
    }
}

#[test]
fn slice_out_of_bounds_end() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let result = view.slice(0, 4);
    assert!(result.is_err());
    
    if let Err(TensorError::CoordsOutOfBounds { rank, provided, max }) = result {
        assert_eq!(rank, 1);
        assert_eq!(provided, 3);
        assert_eq!(max, 2);
    }
}

#[test]
fn slice_invalid_range() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let result = view.slice(2, 1);
    assert!(result.is_err());
}

#[test]
fn slice_empty_range() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let view = tensor.view();
    
    let result = view.slice(1, 1);
    assert!(result.is_err());
}

#[test]
fn slice_3d_tensor() {
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
        vec![2, 2, 2]
    );
    let view = tensor.view();
    
    // Slice along last dimension (columns)
    let slice = view.slice(0, 1).unwrap();
    assert_eq!(slice.shape, vec![2, 2, 1]);
    
    // Should get first column of each 2x2 slice
    assert_eq!(slice.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(slice.get(&[0, 1, 0]).unwrap(), 3.0);
    assert_eq!(slice.get(&[1, 0, 0]).unwrap(), 5.0);
    assert_eq!(slice.get(&[1, 1, 0]).unwrap(), 7.0);
}

#[test]
fn slice_preserves_strides() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = tensor.view();
    
    let slice = view.slice(1, 3).unwrap();
    
    // Original strides should be preserved for 2D+
    assert_eq!(slice.strides, view.strides);
    assert_eq!(slice.strides, vec![3, 1]);
}

#[test]
fn square_vector() {
    let tensor = Tensor::new(vec![2.0, -3.0, 4.0], vec![3]);
    let view = tensor.view();
    let result = view.square().unwrap();
    
    assert_eq!(result.elements, vec![4.0, 9.0, 16.0]);
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn square_matrix() {
    let tensor = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
    let view = tensor.view();
    let result = view.square().unwrap();
    
    assert_eq!(result.elements, vec![1.0, 4.0, 9.0, 16.0]);
    assert_eq!(result.shape, vec![2, 2]);
}

#[test]
fn square_single_element() {
    let tensor = Tensor::new(vec![-5.0], vec![1]);
    let view = tensor.view();
    let result = view.square().unwrap();
    
    assert_eq!(result.elements, vec![25.0]);
    assert_eq!(result.shape, vec![1]);
}

#[test]
fn mse_perfect_match() {
    let predictions = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    
    let pred_view = predictions.view();
    let target_view = targets.view();
    
    let mse = pred_view.mse(&target_view).unwrap();
    assert_eq!(mse, 0.0);
}

#[test]
fn mse_simple_errors() {
    let predictions = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = Tensor::new(vec![2.0, 4.0, 5.0], vec![3]);
    
    let pred_view = predictions.view();
    let target_view = targets.view();
    
    // errors: [-1.0, -2.0, -2.0]
    // squared: [1.0, 4.0, 4.0] 
    // mean: 9.0 / 3 = 3.0
    let mse = pred_view.mse(&target_view).unwrap();
    assert_eq!(mse, 3.0);
}

#[test]
fn mse_matrix() {
    let predictions = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let targets = Tensor::new(vec![2.0, 2.0, 5.0, 4.0], vec![2, 2]);
    
    let pred_view = predictions.view();
    let target_view = targets.view();
    
    // errors: [-1.0, 0.0, -2.0, 0.0]
    // squared: [1.0, 0.0, 4.0, 0.0]
    // mean: 5.0 / 4 = 1.25
    let mse = pred_view.mse(&target_view).unwrap();
    assert_eq!(mse, 1.25);
}

#[test]
fn mse_shape_mismatch() {
    let predictions = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = Tensor::new(vec![1.0, 2.0], vec![2]);

    let pred_view = predictions.view();
    let target_view = targets.view();

    let result = pred_view.mse(&target_view);
    assert!(result.is_err());

    if let Err(TensorError::ShapeMismatch { provided, expected }) = result {
        assert_eq!(provided, vec![2]);
        assert_eq!(expected, vec![3]);
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

// ReLU Tests

#[test]
fn relu_basic_vector() {
    let input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let result = input.view().relu().unwrap();

    assert_eq!(result.elements, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    assert_eq!(result.shape, vec![5]);
}

#[test]
fn relu_matrix() {
    let input = Tensor::new(
        vec![
            -3.0, 2.0, -1.0,
            5.0, -2.0, 0.0
        ],
        vec![2, 3]
    );
    let result = input.view().relu().unwrap();

    assert_eq!(result.elements, vec![0.0, 2.0, 0.0, 5.0, 0.0, 0.0]);
    assert_eq!(result.shape, vec![2, 3]);
}

// Softmax Tests

#[test]
fn softmax_basic_vector() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = input.view().softmax().unwrap();

    // Verify all elements are between 0 and 1
    for &val in &result.elements {
        assert!(val >= 0.0 && val <= 1.0);
    }

    // Verify sum equals 1.0 (with floating-point tolerance)
    let sum: f32 = result.elements.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify monotonicity: larger inputs should have higher probabilities
    assert!(result.elements[0] < result.elements[1]);
    assert!(result.elements[1] < result.elements[2]);
}

#[test]
fn softmax_large_values_stability() {
    // Test numerical stability with large values
    let input = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![3]);
    let result = input.view().softmax().unwrap();

    // Verify no NaN or infinity
    for &val in &result.elements {
        assert!(val.is_finite());
        assert!(!val.is_nan());
    }

    // Verify sum equals 1.0 (with floating-point tolerance)
    let sum: f32 = result.elements.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify all elements are between 0 and 1
    for &val in &result.elements {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn softmax_batch_2x3() {
    // Test batch softmax with 2 samples, 3 classes each
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0,  // Sample 1
            4.0, 5.0, 6.0,  // Sample 2
        ],
        vec![2, 3]
    );
    let result = input.view().softmax().unwrap();

    assert_eq!(result.shape, vec![2, 3]);

    // Each row should sum to 1.0
    let row0_sum = result.get_nd(&[0, 0]).unwrap()
                 + result.get_nd(&[0, 1]).unwrap()
                 + result.get_nd(&[0, 2]).unwrap();
    assert!((row0_sum - 1.0).abs() < 1e-6);

    let row1_sum = result.get_nd(&[1, 0]).unwrap()
                 + result.get_nd(&[1, 1]).unwrap()
                 + result.get_nd(&[1, 2]).unwrap();
    assert!((row1_sum - 1.0).abs() < 1e-6);

    // All values should be between 0 and 1
    for i in 0..2 {
        for j in 0..3 {
            let val = result.get_nd(&[i, j]).unwrap();
            assert!(val >= 0.0 && val <= 1.0);
            assert!(val.is_finite());
        }
    }

    // Within each row, larger inputs should have higher probabilities
    assert!(result.get_nd(&[0, 0]).unwrap() < result.get_nd(&[0, 1]).unwrap());
    assert!(result.get_nd(&[0, 1]).unwrap() < result.get_nd(&[0, 2]).unwrap());
    assert!(result.get_nd(&[1, 0]).unwrap() < result.get_nd(&[1, 1]).unwrap());
    assert!(result.get_nd(&[1, 1]).unwrap() < result.get_nd(&[1, 2]).unwrap());
}

#[test]
fn softmax_batch_single_sample() {
    // Test that batch softmax with 1 sample gives same result as vector softmax
    let input_batch = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let input_vec = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let result_batch = input_batch.view().softmax().unwrap();
    let result_vec = input_vec.view().softmax().unwrap();

    assert_eq!(result_batch.shape, vec![1, 3]);
    assert_eq!(result_vec.shape, vec![3]);

    // Values should match
    for i in 0..3 {
        let batch_val = result_batch.get_nd(&[0, i]).unwrap();
        let vec_val = result_vec.get_nd(&[i]).unwrap();
        assert!((batch_val - vec_val).abs() < 1e-6);
    }
}

#[test]
fn softmax_batch_large_values_stability() {
    // Test numerical stability with large values in batch mode
    let input = Tensor::new(
        vec![
            1000.0, 1001.0, 1002.0,
            2000.0, 2001.0, 2002.0,
        ],
        vec![2, 3]
    );
    let result = input.view().softmax().unwrap();

    // Verify no NaN or infinity
    for i in 0..2 {
        for j in 0..3 {
            let val = result.get_nd(&[i, j]).unwrap();
            assert!(val.is_finite());
            assert!(!val.is_nan());
        }
    }

    // Each row should sum to 1.0
    for i in 0..2 {
        let row_sum = (0..3).map(|j| result.get_nd(&[i, j]).unwrap()).sum::<f32>();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn softmax_batch_multiple_sizes() {
    // Test different batch sizes
    let test_cases = vec![
        (vec![1, 2], "1x2"),
        (vec![3, 4], "3x4"),
        (vec![5, 10], "5x10"),
    ];

    for (shape, label) in test_cases {
        let size = shape[0] * shape[1];
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let input = Tensor::new(data, shape.clone());
        let result = input.view().softmax().unwrap();

        assert_eq!(result.shape, shape, "Failed for {}", label);

        // Each row should sum to 1.0
        for i in 0..shape[0] {
            let row_sum: f32 = (0..shape[1])
                .map(|j| result.get_nd(&[i, j]).unwrap())
                .sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row sum failed for {}", label);
        }
    }
}

#[test]
fn softmax_batch_equal_values() {
    // When all values in a row are equal, softmax should give uniform distribution
    let input = Tensor::new(
        vec![
            5.0, 5.0, 5.0,
            2.0, 2.0, 2.0,
        ],
        vec![2, 3]
    );
    let result = input.view().softmax().unwrap();

    // Each element should be approximately 1/3
    let expected = 1.0 / 3.0;
    for i in 0..2 {
        for j in 0..3 {
            let val = result.get_nd(&[i, j]).unwrap();
            assert!((val - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn softmax_batch_independent_rows() {
    // Verify that softmax is applied independently to each row
    // Row 1: [1, 2, 3] should give same result as standalone vector
    // Row 2: [3, 2, 1] should give same result as standalone vector

    let batch = Tensor::new(
        vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        vec![2, 3]
    );
    let vec1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let vec2 = Tensor::new(vec![3.0, 2.0, 1.0], vec![3]);

    let batch_result = batch.view().softmax().unwrap();
    let vec1_result = vec1.view().softmax().unwrap();
    let vec2_result = vec2.view().softmax().unwrap();

    // Row 0 of batch should match vec1
    for j in 0..3 {
        let batch_val = batch_result.get_nd(&[0, j]).unwrap();
        let vec_val = vec1_result.get_nd(&[j]).unwrap();
        assert!((batch_val - vec_val).abs() < 1e-6);
    }

    // Row 1 of batch should match vec2
    for j in 0..3 {
        let batch_val = batch_result.get_nd(&[1, j]).unwrap();
        let vec_val = vec2_result.get_nd(&[j]).unwrap();
        assert!((batch_val - vec_val).abs() < 1e-6);
    }
}

#[test]
fn softmax_rejects_3d_tensor() {
    let input = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2]
    );
    let result = input.view().softmax();

    assert!(result.is_err());

    if let Err(TensorError::NotVectorError(rank)) = result {
        assert_eq!(rank, 3);
    } else {
        panic!("Expected NotVectorError");
    }
}
