import numpy as np

from PIL import Image
from typing import Any
from typing import List
from typing import Tuple
from skimage.filters import gaussian
from skimage.filters import unsharp_mask


def is_gray(arr: np.ndarray) -> bool:
    if isinstance(arr, np.ndarray):
        return arr.shape[-1] == 1
    if len(arr.shape) == 3:
        return arr.shape[0] == 1
    return arr.shape[1] == 1


def min_max_normalize(arr: np.ndarray, *, global_norm: bool = True) -> np.ndarray:
    eps = 1.0e-8
    if global_norm:
        arr_min, arr_max = arr.min().item(), arr.max().item()
        return (arr - arr_min) / max(eps, arr_max - arr_min)
    arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
    diff = np.maximum(eps, arr_max - arr_min)
    return (arr - arr_min) / diff


def imagenet_normalize(arr: np.ndarray) -> np.ndarray:
    mean_gray, std_gray = [0.485], [0.229]
    mean_rgb, std_rgb = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    np_constructor = lambda inp: np.array(inp, dtype=np.float32).reshape([1, 1, -1])
    if is_gray(arr):
        mean, std = map(np_constructor, [mean_gray, std_gray])
    else:
        mean, std = map(np_constructor, [mean_rgb, std_rgb])
    return (arr - mean) / std


def to_uint8(normalized: np.ndarray) -> np.ndarray:
    return (np.clip(normalized * 255.0, 0.0, 255.0)).astype(np.uint8)


def naive_cutout(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 4:
        img = img[..., :3] * img[..., -1:]
    return to_uint8(np.concatenate([img, alpha[..., None]], axis=2))


def cutout(
    img: np.ndarray,
    alpha: np.ndarray,
    smooth: int = 4,
    tight: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha_im = Image.fromarray(min_max_normalize(alpha))
    alpha_im = alpha_im.resize((img.shape[1], img.shape[0]), Image.NEAREST)
    alpha = gaussian(np.array(alpha_im), smooth)
    alpha = unsharp_mask(alpha, smooth, smooth * tight)
    alpha = min_max_normalize(alpha)
    rgba = naive_cutout(img, alpha)
    return alpha, rgba


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