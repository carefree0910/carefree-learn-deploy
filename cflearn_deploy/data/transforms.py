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


class ToGray:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.dot([[0.2989], [0.587], [0.114]])


class ToNCHW:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.transpose([0, 3, 1, 2])


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


class ImagenetPreprocess:
    def __init__(self):  # type: ignore
        self.to_gray = ToGray()
        self.transform = Compose([ImagenetNormalize(), ToNCHW()])

    def __call__(self, img: np.ndarray, is_gray: bool) -> np.ndarray:
        if is_gray:
            img = self.to_gray(img)
        img = self.transform(img)
        return img.astype(np.float32)
