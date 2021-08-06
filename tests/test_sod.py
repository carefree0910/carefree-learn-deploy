import os
import unittest

import numpy as np

from cflearn_deploy.sod.core import SOD


current_folder = os.path.dirname(__file__)


class TestSOD(unittest.TestCase):
    def test_get_alpha(self) -> None:
        sod = SOD(os.path.join(current_folder, "models", "test.onnx"))
        sod._get_alpha(np.random.random([320, 320, 3]))


if __name__ == "__main__":
    unittest.main()
