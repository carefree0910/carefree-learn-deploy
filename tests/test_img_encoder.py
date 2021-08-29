import os
import unittest

import numpy as np

from cflearn_deploy.toolkit import np_to_bytes
from cflearn_deploy.encoder.core import ImageEncoder


current_folder = os.path.dirname(__file__)


class TestImageEncoder(unittest.TestCase):
    def test_img_encoder(self) -> None:
        test_src = np.random.random([224, 224, 3]).astype(np.float32)
        test_src2 = np.random.random([320, 320, 3]).astype(np.float32)
        encoder = ImageEncoder(os.path.join(current_folder, "models", "cbir_test.onnx"))
        encoder._get_code(test_src)
        with self.assertRaises(Exception):
            encoder._get_code(test_src2)
        encoder.run(np_to_bytes(test_src))
        with self.assertRaises(Exception):
            encoder.run(np_to_bytes(test_src2))


if __name__ == "__main__":
    unittest.main()
