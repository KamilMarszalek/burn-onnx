use super::broadcast_helpers::channel_broadcast_shape;
use super::prelude::*;
use onnx_ir::node::image_scaler::ImageScalerNode;

impl NodeCodegen for ImageScalerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // ImageScaler applies Y[n,c,h,w] = X[n,c,h,w] * scale + bias[c].
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);

        let scale = self.config.scale.unwrap_or(1.0);
        let bias = self.config.bias.as_deref();

        let rank = self.inputs[0].ty.rank();

        let bias_code = if let Some(bias) = bias {
            let channels = bias.len();

            // Broadcast per-channel bias over the batch and spatial dimensions.
            let affine_shape = channel_broadcast_shape(rank, quote! { #channels });

            quote! {
                let bias_tensor = Tensor::<1>::from_data(
                    [#(#bias),*],
                    (&self.device, #output.dtype()),
                )
                .reshape(#affine_shape);
                let #output = #output.add(bias_tensor);
            }
        } else {
            quote! {}
        };

        quote! {
            let #output = #input.mul_scalar(#scale);
            #bias_code
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::image_scaler::{ImageScalerConfig, ImageScalerNodeBuilder};

    #[test]
    fn test_image_scaler_scale_and_bias() {
        let node = ImageScalerNodeBuilder::new("image_scaler")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(ImageScalerConfig::new(Some(2.0), Some(vec![0.1, 0.2, 0.3])))
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = input.mul_scalar(2f32);
            let bias_tensor = Tensor::<
                1,
            >::from_data([0.1f32, 0.2f32, 0.3f32], (&self.device, output.dtype()))
                .reshape([1usize, 3usize, 1usize, 1usize]);
            let output = output.add(bias_tensor);
            output
        }
        ");
    }

    #[test]
    fn test_image_scaler_scale_only() {
        let node = ImageScalerNodeBuilder::new("image_scaler")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(ImageScalerConfig::new(Some(2.0), None))
            .build();

        let code = codegen_forward_default(&node);

        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = input.mul_scalar(2f32);
            output
        }
        ");
    }

    #[test]
    fn test_image_scaler_defaults() {
        let node = ImageScalerNodeBuilder::new("image_scaler")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(ImageScalerConfig::new(None, None))
            .build();

        let code = codegen_forward_default(&node);

        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = input.mul_scalar(1f32);
            output
        }
        ");
    }
}
