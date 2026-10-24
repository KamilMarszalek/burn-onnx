//! # ImageScaler
//!
//! Applies per-channel scaling and bias to an image tensor.
//! Y = X * scale + bias
//!
//! Input shape: [N, C, H, W]

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for the ImageScaler operation.
#[derive(Debug, Clone, Default, new)]
pub struct ImageScalerConfig {
    /// Scale factor applied to all elements. Defaults to 1.0.
    pub scale: Option<f32>,

    /// Optional per-channel bias with length equal to C.
    pub bias: Option<Vec<f32>>,
}

/// Node representation for Image Scaler operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ImageScalerNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ImageScalerConfig,
}

pub(crate) struct ImageScalerProcessor;

impl NodeProcessor for ImageScalerProcessor {
    type Config = ImageScalerConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "tensor(float16 | float | double)".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };
        if !matches!(tensor.dtype, DType::F16 | DType::F32 | DType::F64) {
            return Err(ProcessError::TypeMismatch {
                expected: "tensor(float16 | float | double)".to_string(),
                actual: format!("tensor({:?})", tensor.dtype),
            });
        }

        if tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "ImageScaler expects input tensor of rank 4 (N x C x H x W), got rank {}",
                tensor.rank
            )));
        }

        if let Some(bias) = node
            .attrs
            .get("bias")
            .map(|value| value.clone().into_f32s())
            .transpose()?
            && let Some(shape) = &tensor.static_shape
            && let Some(Some(channels)) = shape.get(1)
            && bias.len() != *channels
        {
            return Err(ProcessError::InvalidAttribute {
                name: "bias".to_string(),
                reason: format!(
                    "bias length must match channel dimension C, got {} and {}",
                    bias.len(),
                    channels
                ),
            });
        }

        node.outputs[0].ty = node.inputs[0].ty.clone();
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let scale: Option<f32> = node
            .attrs
            .get("scale")
            .map(|value| value.clone().into_f32())
            .transpose()?;
        let bias: Option<Vec<f32>> = node
            .attrs
            .get("bias")
            .map(|value| value.clone().into_f32s())
            .transpose()?;
        Ok(ImageScalerConfig::new(scale, bias))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::ImageScaler(ImageScalerNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn make_node(
        scale: Option<f32>,
        bias: Option<Vec<f32>>,
        dtype: DType,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> RawNode {
        let builder = TestNodeBuilder::new(NodeType::ImageScaler, "test_image_scaler");

        let builder = match dtype {
            DType::F16 => builder.input_tensor_f16("X", rank, static_shape),
            DType::F32 => builder.input_tensor_f32("X", rank, static_shape),
            DType::F64 => builder.input_tensor_f64("X", rank, static_shape),
            DType::I32 => builder.input_tensor_i32("X", rank, static_shape),
            other => panic!("unsupported dtype in test helper: {other:?}"),
        };

        let mut builder = builder.output_tensor_f32("Y", rank, None);

        if let Some(scale) = scale {
            builder = builder.attr_float("scale", scale);
        }

        if let Some(bias) = bias {
            builder = builder.attr_floats("bias", bias);
        }

        builder.build()
    }

    #[test]
    fn test_infer_types_preserves_dtype_and_shape() {
        let processor = ImageScalerProcessor;
        let prefs = OutputPreferences::new();

        for dtype in [DType::F16, DType::F32, DType::F64] {
            let mut node = make_node(
                Some(2.0),
                Some(vec![0.1, 0.2, 0.3]),
                dtype,
                4,
                Some(vec![2, 3, 8, 8]),
            );

            let expected = node.inputs[0].ty.clone();

            processor.infer_types(&mut node, 1, &prefs).unwrap();

            assert_eq!(node.outputs[0].ty, expected);
        }
    }

    #[test]
    fn test_infer_types_rejects_invalid_dtype() {
        let mut node = make_node(None, None, DType::I32, 4, Some(vec![1, 3, 8, 8]));

        let err = ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_infer_types_rejects_non_tensor_input() {
        let mut node = TestNodeBuilder::new(NodeType::ImageScaler, "test_image_scaler")
            .input_scalar_f32("X")
            .output_tensor_f32("Y", 4, None)
            .build();

        let err = ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_infer_types_rejects_invalid_rank() {
        let mut node = make_node(None, None, DType::F32, 3, Some(vec![1, 3, 8]));

        let err = ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(
            err,
            ProcessError::Custom(ref message)
                if message.contains("rank 4")
        ));
    }

    #[test]
    fn test_infer_types_rejects_bias_length_mismatch() {
        let mut node = make_node(
            Some(2.0),
            Some(vec![0.1, 0.2]),
            DType::F32,
            4,
            Some(vec![1, 3, 8, 8]),
        );

        let err = ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(
            err,
            ProcessError::InvalidAttribute { ref name, .. }
                if name == "bias"
        ));
    }

    #[test]
    fn test_infer_types_unknown_channels() {
        let mut node = TestNodeBuilder::new(NodeType::ImageScaler, "test_image_scaler")
            .add_input(
                "X",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 4,
                    static_shape: Some(vec![Some(1), None, Some(8), Some(8)]),
                }),
            )
            .output_tensor_f32("Y", 4, None)
            .attr_floats("bias", vec![0.1, 0.2, 0.3])
            .build();

        ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::new())
            .unwrap();

        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn test_extract_config_both_attrs() {
        let node = make_node(Some(2.0), Some(vec![0.1, 0.2, 0.3]), DType::F32, 4, None);

        let config = ImageScalerProcessor.extract_config(&node, 1).unwrap();

        assert_eq!(config.scale, Some(2.0));
        assert_eq!(config.bias, Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_extract_config_no_attrs() {
        let node = make_node(None, None, DType::F32, 4, None);

        let config = ImageScalerProcessor.extract_config(&node, 1).unwrap();

        assert_eq!(config.scale, None);
        assert_eq!(config.bias, None);
    }

    #[test]
    fn test_extract_config_invalid_scale_type() {
        let node = TestNodeBuilder::new(NodeType::ImageScaler, "test_image_scaler")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scale", vec![2.0])
            .build();

        let err = ImageScalerProcessor.extract_config(&node, 1).unwrap_err();

        assert!(matches!(
            err,
            ProcessError::Custom(ref message)
                if message.contains("expected a Float32 attribute")
        ));
    }

    #[test]
    fn test_extract_config_invalid_bias_type() {
        let node = TestNodeBuilder::new(NodeType::ImageScaler, "test_image_scaler")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("bias", 1.0)
            .build();

        let err = ImageScalerProcessor.extract_config(&node, 1).unwrap_err();

        assert!(matches!(
            err,
            ProcessError::Custom(ref message)
                if message.contains("expected a Float32s attribute")
        ));
    }

    #[test]
    fn test_build_node() {
        let raw_node = make_node(
            Some(2.0),
            Some(vec![0.1, 0.2, 0.3]),
            DType::F32,
            4,
            Some(vec![1, 3, 8, 8]),
        );

        let node = ImageScalerProcessor.build_node(raw_node, 1).unwrap();

        match node {
            Node::ImageScaler(image_scaler) => {
                assert_eq!(image_scaler.name, "test_image_scaler");
                assert_eq!(image_scaler.inputs.len(), 1);
                assert_eq!(image_scaler.outputs.len(), 1);
                assert_eq!(image_scaler.inputs[0].name, "X");
                assert_eq!(image_scaler.outputs[0].name, "Y");
                assert_eq!(image_scaler.config.scale, Some(2.0));
                assert_eq!(image_scaler.config.bias, Some(vec![0.1, 0.2, 0.3]));
            }
            other => panic!("expected ImageScaler node, got {other:?}"),
        }
    }
}
