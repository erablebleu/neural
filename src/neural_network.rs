use crate::{math::{sigmoid, sigmoid_deriv}, matrix::Matrix, seeded_random::SeededRandom};

struct Layer {
    weights: Matrix,
    bias: Matrix,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new<I: IntoIterator<Item = usize>>(input_size: usize, hidden_layers_sizes: I, output_size: usize) -> Self {
        let mut random = SeededRandom::new(3615);
        let mut layers = vec![];
        let mut s = input_size;

        for hidden_layer_size in hidden_layers_sizes.into_iter() {
            layers.push(Layer {
                weights: Matrix::from_function(hidden_layer_size, s, |_, _| random.next_weight()),
                bias: Matrix::zero(hidden_layer_size, 1),
            });

            s = hidden_layer_size;
        }

        layers.push(Layer {
            weights: Matrix::from_function(output_size, s, |_, _| random.next_weight()),
            bias: Matrix::zero(output_size, 1),
        });

        Self {
            layers,
        }
    }

    fn compute(&self, set: &Vec<f32>) -> Matrix {
        let mut result = Matrix::from_values(set.len(), 1, set.iter().cloned());

        for layer in self.layers.iter() {
            result = (&layer.weights * result + &layer.bias).map(sigmoid);
        }

        result
    }

    fn train(&mut self, sets: &[(Vec<f32>, Vec<f32>)], epochs: usize) {
        let learning_rate = 1.0;
        let activation_function = sigmoid;
        let activation_derivative = sigmoid_deriv;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for set in sets {
                let mut result = Matrix::from_values(set.0.len(), 1, set.0.iter().cloned());
                let mut layer_results = vec![result.clone()];

                /* forward */
                for layer in self.layers.iter() {
                    result = (&layer.weights * result + &layer.bias).map(activation_function);
                    layer_results.push(result.clone());
                }

                /* back propagation */
                let target: Matrix = Matrix::from_values(set.1.len(), 1, set.1.iter().cloned());
                let mut errors = &target - &result;
                let mut gradients = result.map(activation_derivative);

                let costs = errors.map(|v| v * v);
                total_loss += costs.sum() / costs.len() as f32;

                for layer_index in (0..self.layers.len()).rev() {
                    let layer = &mut self.layers[layer_index];
                    let layer_result = &layer_results[layer_index];

                    gradients = gradients.elementwise_multiply(&errors).map(|v| v * learning_rate);

                    layer.weights = &layer.weights + &gradients * layer_result.transpose();
                    layer.bias = &layer.bias + gradients;

                    errors = layer.weights.transpose() * errors;
                    gradients = layer_result.map(activation_derivative);
                }
            }

            let loss = total_loss / sets.len() as f32;

            if epoch % 50 == 0 {
                println!("epoch {:2}: loss: {}", epoch, loss);
            }
        }
    }

}

#[cfg(test)] 
mod tests {
    use super::*;

    #[test]
    fn test_nn_xor() {
        let mut nn = NeuralNetwork::new(2, [4], 1);

        let sets = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        nn.train(&sets, 5000);

        for set in sets {
            let r = nn.compute(&set.0);
            println!("[{}, {}] -> {}", set.0[0], set.0[1], r.get(0, 0));
        }
    }
}