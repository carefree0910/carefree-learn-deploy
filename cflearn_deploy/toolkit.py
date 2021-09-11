import io

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from skimage.filters import gaussian
from skimage.filters import unsharp_mask

from .constants import WARNING_PREFIX


def is_gray(arr: np.ndarray) -> bool:
    return arr.shape[-1] == 1


def sigmoid(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-arr))


def softmax(arr: np.ndarray) -> np.ndarray:
    logits = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def min_max_normalize(arr: np.ndarray, *, global_norm: bool = True) -> np.ndarray:
    eps = 1.0e-8
    if global_norm:
        arr_min, arr_max = arr.min().item(), arr.max().item()
        return (arr - arr_min) / max(eps, arr_max - arr_min)
    arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
    diff = np.maximum(eps, arr_max - arr_min)
    return (arr - arr_min) / diff


def quantile_normalize(
    arr: np.ndarray,
    *,
    q: float = 0.01,
    global_norm: bool = True,
) -> np.ndarray:
    eps = 1.0e-8
    if global_norm:
        arr_min = np.quantile(arr, q).item()
        arr_max = np.quantile(arr, 1.0 - q).item()
        diff = max(eps, arr_max - arr_min)
    else:
        arr_min = np.quantile(arr, q, axis=0)
        arr_max = np.quantile(arr, 1.0 - q, axis=0)
        diff = np.maximum(eps, arr_max - arr_min)
    arr = np.clip(arr, arr_min, arr_max)
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


def to_uint8(normalized_img: np.ndarray) -> np.ndarray:
    return (np.clip(normalized_img * 255.0, 0.0, 255.0)).astype(np.uint8)


def naive_cutout(normalized_img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if normalized_img.shape[-1] == 4:
        normalized_img = normalized_img[..., :3] * normalized_img[..., -1:]
    return to_uint8(np.concatenate([normalized_img, alpha[..., None]], axis=2))


def alpha_align(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_im = Image.fromarray(min_max_normalize(alpha))
    size = img.shape[1], img.shape[0]
    alpha = np.array(alpha_im.resize(size, Image.LANCZOS))
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def cutout(
    normalized_img: np.ndarray,
    alpha: np.ndarray,
    smooth: int = 0,
    tight: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = alpha_align(normalized_img, alpha)
    if smooth > 0:
        alpha = gaussian(alpha, smooth)
        alpha = unsharp_mask(alpha, smooth, smooth * tight)
    alpha = quantile_normalize(alpha)
    rgba = naive_cutout(normalized_img, alpha)
    return alpha, rgba


def bytes_to_np(img_bytes: bytes, *, mode: str) -> np.ndarray:
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert(mode))
    return img.astype(np.float32) / 255.0


def np_to_bytes(img_arr: np.ndarray) -> bytes:
    if img_arr.dtype != np.uint8:
        img_arr = to_uint8(img_arr)
    bytes_io = io.BytesIO()
    Image.fromarray(img_arr).save(bytes_io, format="PNG")
    return bytes_io.getvalue()


def register_core(
    name: str,
    global_dict: Dict[str, type],
    *,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
) -> Callable[[Type], Type]:
    def _register(cls: Type) -> Type:
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(
                f"{WARNING_PREFIX}'{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


T = TypeVar("T")


class WithRegister(Generic[T]):
    d: Dict[str, Type[T]]
    __identifier__: str

    @classmethod
    def get(cls, name: str) -> Type[T]:
        return cls.d[name]

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> T:
        return cls.get(name)(**config)  # type: ignore

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)

    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d[name], cls)
