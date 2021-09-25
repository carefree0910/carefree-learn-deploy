import numpy as np

from ..toolkit import to_uint8
from ..toolkit import min_max_normalize
from ..protocol import ONNXModelProtocol
from ..constants import PREDICTIONS_KEY


@ONNXModelProtocol.register("style_gan")
class StyleGAN(ONNXModelProtocol):
    def _generate(self) -> np.ndarray:
        z = np.random.randn(1, 512).astype(np.float32)
        rgb = self.onnx.run(z)[PREDICTIONS_KEY][0].transpose([1, 2, 0])
        return to_uint8(min_max_normalize(rgb))

    def run(self) -> np.ndarray:  # type: ignore
        return self._generate()


__all__ = [
    "StyleGAN",
]
