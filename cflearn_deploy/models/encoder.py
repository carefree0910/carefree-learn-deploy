import dill

import numpy as np

from typing import List

from ..types import np_dict_type
from ..toolkit import bytes_to_np
from ..protocol import ONNXModelProtocol
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

    def _get_code(self, text: List[str]) -> np_dict_type:
        tokens = self.tokenizer.tokenize(text)
        return self.onnx.run(tokens)

    def run(  # type: ignore
        self,
        text: List[str],
        gray: bool = False,
        no_transform: bool = False,
    ) -> np.ndarray:
        assert not gray and not no_transform
        return self._get_code(text)


@ONNXModelProtocol.register("cbir")
@ONNXModelProtocol.register("image_encoder")
class ImageEncoder(ONNXModelProtocol):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_code(
        self,
        src: np.ndarray,
        gray: bool,
        no_transform: bool,
    ) -> np_dict_type:
        if gray:
            src = src.mean(axis=2, keepdims=True)
        if no_transform:
            transformed = src.transpose([2, 0, 1])[None, ...]
        else:
            transformed = self.transform(src)[None, ...]
        return self.onnx.run(transformed)

    def run(  # type: ignore
        self,
        img_bytes: bytes,
        gray: bool = False,
        no_transform: bool = False,
    ) -> np_dict_type:
        src = bytes_to_np(img_bytes, mode="RGB")
        return self._get_code(src, gray, no_transform)


__all__ = [
    "TextEncoder",
    "ImageEncoder",
]
