import numpy as np

from ..types import np_dict_type
from ..toolkit import softmax
from ..toolkit import bytes_to_np
from ..protocol import ONNXModelProtocol
from ..data.transforms import ToCHW
from ..data.transforms import Compose
from ..data.transforms import ImagenetNormalize


@ONNXModelProtocol.register("clf")
class Clf(ONNXModelProtocol):
    def __init__(self, onnx_path: str):
        super().__init__(onnx_path)
        self.transform = Compose([ImagenetNormalize(), ToCHW()])

    def _get_prob_dict(
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
        logits_dict = self.onnx.run(transformed)
        return {k: softmax(v)[0] for k, v in logits_dict.items()}

    def run(  # type: ignore
        self,
        img_bytes: bytes,
        gray: bool = False,
        no_transform: bool = False,
    ) -> np_dict_type:
        src = bytes_to_np(img_bytes, mode="RGB")
        return self._get_prob_dict(src, gray, no_transform)


__all__ = [
    "Clf",
]
