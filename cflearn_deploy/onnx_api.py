import numpy as np

from typing import Union

from .types import np_dict_type

try:
    from onnxruntime import SessionOptions
    from onnxruntime import InferenceSession
    from onnxruntime import GraphOptimizationLevel
except ImportError:
    SessionOptions = None
    InferenceSession = None
    GraphOptimizationLevel = None


class ONNX:
    def __init__(self, path: str):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = InferenceSession(
            path,
            options,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
            ],
            [
                {},
                {},
                {"device_type": "CPU_FP32"},
            ],
        )
        self.output_names = [node.name for node in self.sess.get_outputs()]

    def run(self, inp: Union[np.ndarray, np_dict_type]) -> np_dict_type:
        if isinstance(inp, np.ndarray):
            inp = {node.name: inp for node in self.sess.get_inputs()}
        return dict(zip(self.output_names, self.sess.run(None, inp)))


__all__ = [
    "ONNX",
]
