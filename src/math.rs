pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn sigmoid_deriv(value: f32) -> f32 {
    value * (1.0 - value)
}