import os
import json
import unittest

import numpy as np

from apis.interface import app
from fastapi.testclient import TestClient
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import _get_img_post_kwargs
from cflearn_deploy.constants import PREDICTIONS_KEY


client = TestClient(app)
current_folder = os.path.dirname(__file__)


class TestAPIs(unittest.TestCase):
    def test_adain(self) -> None:
        img = np.random.random([224, 224, 3])
        onnx_path = os.path.join(current_folder, "models", "adain_test.onnx")
        kwargs = _get_img_post_kwargs(img, img, onnx_path=onnx_path)
        response = client.post("/cv/adain", **kwargs)
        self.assertEqual(response.status_code, 200)
        stylized = bytes_to_np(response.content, mode="RGB")
        self.assertSequenceEqual(stylized.shape, [224, 224, 3])

    def test_clf(self) -> None:
        img = np.random.random([224, 224, 3])
        onnx_path = os.path.join(current_folder, "models", "clf_test.onnx")
        kwargs = _get_img_post_kwargs(img, onnx_path=onnx_path)
        response = client.post("/cv/clf", **kwargs)
        self.assertEqual(response.status_code, 200)
        probabilities = json.loads(response.content)["probabilities"]
        self.assertEqual(len(probabilities[PREDICTIONS_KEY]), 100)

    def test_sod(self) -> None:
        img = np.random.random([320, 320, 3])
        onnx_path = os.path.join(current_folder, "models", "sod_test.onnx")
        kwargs = _get_img_post_kwargs(img, onnx_path=onnx_path)
        response = client.post("/cv/sod", **kwargs)
        self.assertEqual(response.status_code, 200)
        rgba = bytes_to_np(response.content, mode="RGBA")
        self.assertSequenceEqual(rgba.shape, [320, 320, 4])

    def test_cbir(self) -> None:
        img = np.random.random([224, 224, 3])
        onnx_path = os.path.join(current_folder, "models", "cbir_test.onnx")
        kwargs = _get_img_post_kwargs(
            img,
            task="cbir",
            onnx_path=onnx_path,
            skip_faiss=True,
        )
        response = client.post("/cv/cbir", **kwargs)
        self.assertEqual(response.status_code, 200)
        rs = json.loads(response.content)
        self.assertSequenceEqual(rs["files"], [""])
        self.assertSequenceEqual(rs["distances"], [0])


if __name__ == "__main__":
    unittest.main()
