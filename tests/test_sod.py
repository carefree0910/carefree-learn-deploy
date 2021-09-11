import os
import unittest

import numpy as np

from cflearn_deploy.toolkit import np_to_bytes
from cflearn_deploy.models.sod import SOD


current_folder = os.path.dirname(__file__)


class TestSOD(unittest.TestCase):
    def test_sod(self) -> None:
        test_src = np.random.random([320, 320, 3]).astype(np.float32)
        test_src2 = np.random.random([224, 224, 3]).astype(np.float32)
        sod = SOD(os.path.join(current_folder, "models", "sod_test.onnx"))
        sod._get_alpha(test_src)
        with self.assertRaises(Exception):
            sod._get_alpha(test_src2)
        sod.run(np_to_bytes(test_src))
        with self.assertRaises(Exception):
            sod.run(np_to_bytes(test_src2))


if __name__ == "__main__":
    unittest.main()
