import os
import unittest

import numpy as np

from cflearn_deploy.sod.core import SOD


current_folder = os.path.dirname(__file__)


class TestSOD(unittest.TestCase):
    def test_get_cutout(self) -> None:
        sod = SOD(os.path.join(current_folder, "models", "test.onnx"))
        sod._get_alpha(np.random.random([320, 320, 3]))
        src_path = os.path.join(current_folder, "data", "pytorch.png")
        tgt_path = os.path.join(current_folder, "data", "pytorch_cutout.png")
        sod.generate_cutout(src_path, tgt_path)


if __name__ == "__main__":
    unittest.main()
