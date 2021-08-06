import os

import numpy as np

from typing import Tuple
from typing import Optional
from skimage import io
from onnxruntime import InferenceSession

from .data import RescaleT
from .data import ToNormalizedArray
from ..toolkit import cutout
from ..toolkit import Compose
from ..constants import INPUT_KEY


def export(rgba: np.ndarray, tgt_path: Optional[str]) -> None:
    if tgt_path is not None:
        folder = os.path.split(tgt_path)[0]
        os.makedirs(folder, exist_ok=True)
        io.imsave(tgt_path, rgba)


class SOD:
    def __init__(
        self,
        onnx_path: str,
        rescale_size: int = 320,
    ):
        self.ort_session = InferenceSession(onnx_path)
        self.transform = Compose([RescaleT(rescale_size), ToNormalizedArray()])

    def _get_alpha(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform({INPUT_KEY: src})[INPUT_KEY][None, ...]
        ort_inputs = {node.name: transformed for node in self.ort_session.get_inputs()}
        logits = self.ort_session.run(None, ort_inputs)[0][0][0]
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def generate_cutout(
        self,
        src_path: str,
        tgt_path: Optional[str] = None,
        *,
        smooth: int = 16,
        tight: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = io.imread(src_path)
        img = img.astype(np.float32) / 255.0
        alpha = self._get_alpha(img)
        alpha, rgba = cutout(img, alpha, smooth, tight)
        export(rgba, tgt_path)
        return alpha, rgba


__all__ = [
    "SOD",
]
