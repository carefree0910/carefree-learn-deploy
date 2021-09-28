import numpy as np

from ..toolkit import to_uint8
from ..toolkit import bytes_to_np
from ..protocol import ONNXModelProtocol
from ..constants import PREDICTIONS_KEY


@ONNXModelProtocol.register("cycle_gan")
class CycleGANStylizer(ONNXModelProtocol):
    def _get_stylized(self, content: np.ndarray) -> np.ndarray:
        content = content.transpose([2, 0, 1])[None, ...]
        content = 2.0 * content - 1.0
        stylized = self.onnx.run({"input": content})[PREDICTIONS_KEY][0]
        stylized = stylized.transpose([1, 2, 0])
        stylized = 0.5 * (stylized + 1.0)
        return to_uint8(stylized)

    def run(self, img_bytes0: bytes) -> np.ndarray:  # type: ignore
        content = bytes_to_np(img_bytes0, mode="RGB")
        return self._get_stylized(content)


__all__ = [
    "CycleGANStylizer",
]