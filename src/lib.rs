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

        let mut idx = 0;
        let mut stride = 1;

        for i in 0..coords.len() {
            idx += stride * coords[i];
            stride *= self.shape[i];
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
}
