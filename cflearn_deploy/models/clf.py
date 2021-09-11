import numpy as np

from ..toolkit import softmax
from ..toolkit import bytes_to_np
from ..protocol import ModelProtocol
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


@ModelProtocol.register("clf")
class Clf(ModelProtocol):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_prob(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform(src)[None, ...]
        logits = self.onnx.run(transformed)[0]
        return softmax(logits)[0]

    def run(self, img_bytes: bytes) -> np.ndarray:  # type: ignore
        src = bytes_to_np(img_bytes, mode="RGB")
        return self._get_prob(src)


__all__ = [
    "Clf",
]
