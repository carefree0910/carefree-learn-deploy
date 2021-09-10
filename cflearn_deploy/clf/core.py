import numpy as np

from ..toolkit import softmax
from ..toolkit import bytes_to_np
from ..onnx_api import ONNX
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


class Clf:
    def __init__(self, onnx_path: str):
        self.onnx = ONNX(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_prob(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform(src)[None, ...]
        logits = self.onnx.run(transformed)[0]
        return softmax(logits)[0]

    def run(self, img_bytes: bytes) -> np.ndarray:
        src = bytes_to_np(img_bytes, mode="RGB")
        return self._get_prob(src)


__all__ = [
    "Clf",
]
