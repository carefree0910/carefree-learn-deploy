import numpy as np

from typing import Any
from typing import List

from ..toolkit import min_max_normalize
from ..toolkit import imagenet_normalize


class ImagenetNormalize:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = min_max_normalize(img)
        img = imagenet_normalize(img)
        return img


class ToCHW:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.transpose([2, 0, 1])


class Compose:
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, inp: Any) -> Any:
        for t in self.transforms:
            inp = t(inp)
        return inp

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
