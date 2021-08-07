import requests

import numpy as np

from typing import Any
from typing import Dict
from requests import Response

from .toolkit import np_to_bytes


def _get_img_post_kwargs(
    img_arr: np.ndarray,
    timeout: int = 8000,
    **kwargs: Any,
) -> Dict[str, Any]:
    img_bytes = np_to_bytes(img_arr)
    return dict(
        files={"img_bytes": ("", img_bytes, "image/png")},
        params=kwargs,
        timeout=timeout,
    )


def post_img_arr(
    img_arr: np.ndarray,
    *,
    uri: str,
    port: str,
    ip: str = "127.0.0.1",
    timeout: int = 8000,
    **kwargs: Any,
) -> Response:
    return requests.post(
        f"http://{ip}:{port}{uri}",
        **_get_img_post_kwargs(img_arr, timeout, **kwargs),
    )
