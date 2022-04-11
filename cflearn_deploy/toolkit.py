import os
import sys
import json
import faiss

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
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
    np_constructor = lambda inp: np.array(inp, dtype=np.float32).reshape([1, 1, 1, -1])
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


def resize_to(normalized_img: np.ndarray, shape: Any) -> Image.Image:
    img = Image.fromarray(to_uint8(normalized_img))
    img = img.resize(shape, Image.LANCZOS)
    return img


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


def get_compatible_name(name: str, version: Tuple[int, int]) -> str:
    version_info = sys.version_info
    need_compatible = False
    if version_info.major > version[0] or version_info.minor >= version[1]:
        need_compatible = True
    if need_compatible:
        name = f"{name}_{version[0]}.{version[1]}"
    return name


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


class IRMixin:
    faiss_info: Dict[str, Any]
    appendix_list: List[str]

    def init_faiss(self, model: str, current_folder: str) -> None:
        self.faiss_info = {}
        for appendix in self.appendix_list:
            json_path = os.path.join(current_folder, f"{model}{appendix}_files.json")
            faiss_path = os.path.join(current_folder, f"{model}{appendix}.index")
            with open(json_path, "r", encoding="utf-8") as f:
                files = json.load(f)
            index = faiss.read_index(faiss_path)
            self.faiss_info[appendix] = {
                "index": index,
                "files": files,
                "num_total": index.ntotal,
            }

    def get_raw_outputs(
        self,
        appendix: str,
        response: Any,
        n_probe: int,
        top_k: int,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        import triton_python_backend_utils as pb_utils

        info = self.faiss_info[appendix]
        index, files, num_total = info["index"], info["files"], info["num_total"]
        index.nprobe = n_probe
        top_k = min(top_k, num_total)

        tensor = pb_utils.get_output_tensor_by_name(response, "predictions")
        codes = tensor.as_numpy()
        all_files, all_distances = [], []
        for code in codes:
            distances, indices = index.search(code[None, ...], top_k)
            indices = indices[0]
            distances = [d for i, d in enumerate(distances[0]) if indices[i] != -1]
            indices = [i for i in indices if i != -1]
            all_files.append([files[i] for i in indices])
            all_distances.append(distances)
        return all_files, all_distances

    def get_outputs(
        self,
        appendix: str,
        response: Any,
        n_probe: int,
        top_k: int,
    ) -> List[Any]:
        import triton_python_backend_utils as pb_utils

        files, distances = self.get_raw_outputs(appendix, response, n_probe, top_k)
        return [
            pb_utils.Tensor("files", np.array(files, np.object)),
            pb_utils.Tensor("distances", np.array(distances, np.float32)),
        ]
