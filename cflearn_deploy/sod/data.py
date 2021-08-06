import numpy as np

from typing import Union
from skimage import transform as sk_transform

from ..types import np_dict_type
from ..toolkit import min_max_normalize
from ..toolkit import imagenet_normalize
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY


def make_new_sample(sample: np_dict_type, img: np.ndarray) -> np_dict_type:
    new_sample = sample.copy()
    new_sample[INPUT_KEY] = img
    new_sample[LABEL_KEY] = None
    return new_sample


class RescaleT:
    def __init__(self, output_size: Union[int, tuple]):
        if isinstance(output_size, int):
            output_size = output_size, output_size
        self.output_size = output_size

    def __call__(self, sample: np_dict_type) -> np_dict_type:
        img = sk_transform.resize(
            (sample[INPUT_KEY][..., :3] * 255.0).astype(np.uint8),
            self.output_size,
            mode="constant",
        )
        img = img.astype(np.float32)
        return make_new_sample(sample, img)


class ToNormalizedArray:
    def __call__(self, sample: np_dict_type) -> np_dict_type:
        img, label = sample[INPUT_KEY], sample[LABEL_KEY]
        img = min_max_normalize(img)
        img = imagenet_normalize(img)
        return make_new_sample(sample, img.transpose([2, 0, 1]))
