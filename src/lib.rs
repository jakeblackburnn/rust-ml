pub struct Tensor {
    elements: Vec<f32>,
    shape: Vec<usize>,
}
 
impl Tensor {

    pub fn new(elements: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor {
            elements, 
            shape,
        }
    }

    pub fn get_nd(&self, coords: &[usize]) -> Result<f32, String> {
        
        // TODO: check that coords are sensical
        if self.shape.len() != coords.len() {
            return Err( String::from("coords are nonsensical") );
        }

        let mut idx = 0;
        let mut stride = 1;

        for dim in (0..coords.len()).rev() {
            // iterate over coordinates from outer -> inner dimensions
            // (row major)

            let dimlen = self.shape[dim]; // current dimension size
            let dimidx = coords[dim];     // coordinate idx along dimension
            
            if dimidx >= dimlen {
                return Err( String::from("coords went out of bounds") );
            }

            idx += stride * dimidx; 
            stride *= dimlen;      
        }

        Ok( self.elements[idx] )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_element() {
        let x = Tensor::new( vec![1.0, 0.0, -1.0], vec![1, 3]);
        assert_eq!( x.get_nd(&[0, 2]), Ok(-1.0) );
    }

    #[test]
    fn fail_get_element() {
        let x = Tensor::new( vec![1.0, 0.0, -1.0], vec![1, 3]);
        assert_eq!( x.get_nd(&[0, 3]), Err(String::from("coords went out of bounds")) );
    }

    #[test]
    fn fail_get_element2() {
        let x = Tensor::new( vec![1.0, 0.0, -1.0], vec![3, 1]);
        assert_eq!( x.get_nd(&[0, 2]), Err(String::from("coords went out of bounds")) );
    }

    #[test]
    fn fail_from_coord_shape() {
        let x = Tensor::new( vec![1.0, 0.0, -1.0], vec![1, 3]);
        assert_eq!( x.get_nd(&[0, 0, 0]), Err(String::from("coords are nonsensical")) );
    }

    #[test]
    fn fail_from_coord_shape1() {
        let x = Tensor::new( vec![1.0, 0.0, -1.0], vec![1, 3]);
        assert_eq!( x.get_nd(&[0]), Err(String::from("coords are nonsensical")) );
    }
}
