import requests

import numpy as np

from typing import Any
from typing import Dict
from requests import Response

from .toolkit import np_to_bytes


prefix = "http://carefree-learn-deploy:80"


def post_json(body: Any, *, uri: str, timeout: int = 8000, **kwargs: Any) -> Response:
    return requests.post(f"{prefix}{uri}", json=body, params=kwargs, timeout=timeout)


def _get_img_post_kwargs(
    *img_arr: np.ndarray,
    timeout: int = 8000,
    **kwargs: Any,
) -> Dict[str, Any]:
    files = {
        f"img_bytes{i}": ("", np_to_bytes(arr), "image/png")
        for i, arr in enumerate(img_arr)
    }
    return dict(files=files, params=kwargs, timeout=timeout)


def post_img_arr(
    *img_arr: np.ndarray,
    uri: str,
    timeout: int = 8000,
    **kwargs: Any,
) -> Response:
    return requests.post(
        f"{prefix}{uri}",
        **_get_img_post_kwargs(*img_arr, timeout=timeout, **kwargs),
    )
