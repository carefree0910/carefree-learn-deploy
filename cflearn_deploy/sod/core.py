import numpy as np

from ..toolkit import cutout
from ..toolkit import sigmoid
from ..toolkit import bytes_to_np
from ..protocol import ModelProtocol
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


@ModelProtocol.register("sod")
class SOD(ModelProtocol):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_alpha(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform(src)[None, ...]
        logits = self.onnx.run(transformed)[0][0][0]
        logits = np.clip(logits, -50.0, 50.0)
        return sigmoid(logits)

    def run(  # type: ignore
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
