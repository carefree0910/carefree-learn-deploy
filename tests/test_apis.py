import os
import json
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
        model_path = os.path.join(current_folder, "models", "sod_test.onnx")
        kwargs = _get_img_post_kwargs(img, model_path=model_path)
        response = client.post("/cv/sod", **kwargs)
        assert response.status_code == 200

    def test_cbir(self) -> None:
        img = np.random.random([224, 224, 3])
        model_path = os.path.join(current_folder, "models", "cbir_test.onnx")
        kwargs = _get_img_post_kwargs(
            img,
            task="cbir",
            model_path=model_path,
            skip_faiss=True,
        )
        response = client.post("/cv/cbir", **kwargs)
        assert response.status_code == 200
        rs = json.loads(response.content)
        self.assertSequenceEqual(rs["files"], [""])
        self.assertSequenceEqual(rs["distances"], [0])


if __name__ == "__main__":
    unittest.main()
