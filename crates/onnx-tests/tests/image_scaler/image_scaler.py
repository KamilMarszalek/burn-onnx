#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

SCALE = 2.0
BIAS = [0.1, 0.2, 0.3]
SHAPE = [1, 3, 2, 2]


def main():
    node = helper.make_node(
        "ImageScaler",
        inputs=["input"],
        outputs=["output"],
        scale=SCALE,
        bias=BIAS,
    )

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, SHAPE
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, SHAPE
    )

    graph = helper.make_graph(
        [node],
        "image_scaler_test",
        [input_info],
        [output_info],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 1)],
    )

    output_path = Path(__file__).with_name("image_scaler.onnx")
    onnx.save(model, output_path)

    # Reference: Y[n, c, h, w] = X[n, c, h, w] * scale + bias[c].
    input_data = np.arange(1, 13, dtype=np.float32).reshape(SHAPE)
    bias = np.array(BIAS, dtype=np.float32)

    expected = input_data * SCALE + bias.reshape(1, 3, 1, 1)

    print("Saved:", output_path)
    print("Input:", input_data)
    print("Expected output:", expected)


if __name__ == "__main__":
    main()
