use crate::include_models;
include_models!(image_scaler);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance};

    #[test]
    fn test_image_scaler() {
        let device = Default::default();
        let model = image_scaler::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/image_scaler.bpk"),
            &device,
        );

        let input = Tensor::<4>::from_floats(
            [[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]],
            &device,
        );

        let output = model.forward(input);

        // Y[n, c, h, w] = X[n, c, h, w] * 2.0 + bias[c],
        // where bias = [0.1, 0.2, 0.3].
        let expected = TensorData::from([[
            [[2.1f32, 4.1], [6.1, 8.1]],
            [[10.2, 12.2], [14.2, 16.2]],
            [[18.3, 20.3], [22.3, 24.3]],
        ]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
