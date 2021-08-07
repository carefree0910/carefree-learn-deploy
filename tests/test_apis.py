import os
import unittest

import numpy as np

from apis.interface import app
from fastapi.testclient import TestClient
from cflearn_deploy.api_utils import _get_img_post_kwargs


client = TestClient(app)
current_folder = os.path.dirname(__file__)


class TestAPIs(unittest.TestCase):
    def test_sod(self) -> None:
        img = np.random.random([320, 320, 3])
        model_path = os.path.join(current_folder, "models", "test.onnx")
        kwargs = _get_img_post_kwargs(img, model_path=model_path)
        response = client.post("/ai/sod", **kwargs)
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
