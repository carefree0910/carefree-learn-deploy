import numpy as np

from typing import Dict
from typing import Union
from onnxruntime import SessionOptions
from onnxruntime import InferenceSession
from onnxruntime import GraphOptimizationLevel


class ONNX:
    def __init__(self, path: str):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = InferenceSession(
            path,
            options,
            [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
            ],
            [
                {},
                {"device_type": "CPU_FP32"},
            ],
        )

    def run(self, inp: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        if isinstance(inp, np.ndarray):
            inp = {node.name: inp for node in self.sess.get_inputs()}
        return self.sess.run(None, inp)


__all__ = [
    "ONNX",
]
