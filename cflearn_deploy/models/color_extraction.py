import numpy as np

from sklearn.cluster import KMeans

from ..toolkit import to_uint8
from ..toolkit import bytes_to_np
from ..protocol import ModelProtocol


@ModelProtocol.register("color_extraction")
class ColorExtraction(ModelProtocol):
    def __init__(self, num_colors: int):
        self.k_means = KMeans(num_colors)

    def run(self, img_bytes: bytes) -> np.ndarray:
        data = bytes_to_np(img_bytes, mode="RGB")
        data = data.reshape(-1, 3)
        self.k_means.fit(data)
        normalized_colors = self.k_means.cluster_centers_
        colors = to_uint8(normalized_colors)
        return colors


__all__ = [
    "ColorExtraction",
]
