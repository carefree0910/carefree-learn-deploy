import dill

import numpy as np

from typing import List

from ..toolkit import bytes_to_np
from ..onnx_api import ONNX
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


class TextEncoder:
    def __init__(self, onnx_path: str, tokenizer_path: str):
        self.onnx = ONNX(onnx_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = dill.load(f)

    def _get_code(self, text: List[str]) -> np.ndarray:
        tokens = self.tokenizer(text)
        return self.onnx.run(tokens)[0][0]

    def run(self, text: List[str]) -> np.ndarray:
        return self._get_code(text)


class ImageEncoder:
    def __init__(self, onnx_path: str):
        self.onnx = ONNX(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_code(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform(src)[None, ...]
        return self.onnx.run(transformed)[0][0]

    def run(self, img_bytes: bytes) -> np.ndarray:
        src = bytes_to_np(img_bytes, mode="RGB")
        return self._get_code(src)


__all__ = [
    "TextEncoder",
    "ImageEncoder",
]
