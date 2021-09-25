import dill

import numpy as np

from typing import List

from ..toolkit import bytes_to_np
from ..protocol import ONNXModelProtocol
from ..constants import LATENT_KEY
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


@ONNXModelProtocol.register("tbir")
@ONNXModelProtocol.register("text_encoder")
class TextEncoder(ONNXModelProtocol):
    def __init__(self, onnx_path: str, tokenizer_path: str):
        super().__init__(onnx_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = dill.load(f)

    def _get_code(self, text: List[str]) -> np.ndarray:
        tokens = self.tokenizer.tokenize(text)
        return next(iter(self.onnx.run(tokens).values()))[0]

    def run(self, text: List[str]) -> np.ndarray:  # type: ignore
        return self._get_code(text)


@ONNXModelProtocol.register("cbir")
@ONNXModelProtocol.register("image_encoder")
class ImageEncoder(ONNXModelProtocol):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_code(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform(src)[None, ...]
        return self.onnx.run(transformed)[LATENT_KEY][0]

    def run(self, img_bytes: bytes) -> np.ndarray:  # type: ignore
        src = bytes_to_np(img_bytes, mode="RGB")
        return self._get_code(src)


__all__ = [
    "TextEncoder",
    "ImageEncoder",
]
