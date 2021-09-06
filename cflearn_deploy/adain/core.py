import numpy as np

from ..toolkit import to_uint8
from ..toolkit import bytes_to_np
from ..onnx_api import ONNX
from ..data.transforms import ToCHW


class AdaINStylizer:
    def __init__(self, onnx_path: str):
        self.onnx = ONNX(onnx_path)
        self.transform = ToCHW()

    def _get_stylized(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        content = self.transform(content)[None, ...]
        style = self.transform(style)[None, ...]
        stylized = self.onnx.run({"input": content, "style": style})[0][0]
        stylized = stylized.transpose([1, 2, 0])
        return to_uint8(stylized)

    def run(self, img_bytes0: bytes, img_bytes1: bytes) -> np.ndarray:
        content = bytes_to_np(img_bytes0, mode="RGB")
        style = bytes_to_np(img_bytes1, mode="RGB")
        return self._get_stylized(content, style)


__all__ = [
    "AdaINStylizer",
]
