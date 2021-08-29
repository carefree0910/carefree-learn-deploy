import numpy as np

from ..toolkit import cutout
from ..toolkit import bytes_to_np
from ..onnx_api import ONNX
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


class SOD:
    def __init__(self, onnx_path: str):
        self.onnx = ONNX(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_alpha(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform(src)[None, ...]
        logits = self.onnx.run(transformed)[0][0][0]
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def run(
        self,
        img_bytes: bytes,
        *,
        smooth: int = 0,
        tight: float = 0.9,
    ) -> np.ndarray:
        src = bytes_to_np(img_bytes, mode="RGB")
        alpha = self._get_alpha(src)
        return cutout(src, alpha, smooth, tight)[1]


__all__ = [
    "SOD",
]
