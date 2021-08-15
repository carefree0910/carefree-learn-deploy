import numpy as np

from .data import RescaleT
from .data import ToNormalizedArray
from ..toolkit import cutout
from ..toolkit import bytes_to_np
from ..toolkit import Compose
from ..onnx_api import ONNX
from ..constants import INPUT_KEY


class SOD:
    def __init__(
        self,
        onnx_path: str,
        rescale_size: int = 320,
    ):
        self.onnx = ONNX(onnx_path)
        self.normalize = ToNormalizedArray()
        self.transform = Compose([RescaleT(rescale_size), self.normalize])

    def _get_alpha(self, src: np.ndarray, rescale: bool = True) -> np.ndarray:
        transform = self.transform if rescale else self.normalize
        transformed = transform({INPUT_KEY: src})[INPUT_KEY][None, ...]  # type: ignore
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
        alpha = self._get_alpha(src, rescale=False)
        return cutout(src, alpha, smooth, tight)[1]


__all__ = [
    "SOD",
]
