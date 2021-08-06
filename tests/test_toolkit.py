import unittest

import numpy as np

from cflearn_deploy.toolkit import cutout
from cflearn_deploy.toolkit import is_gray


class TestToolkit(unittest.TestCase):
    def test_cutout(self) -> None:
        img = np.random.random([320, 320, 3])
        alpha = np.random.random([320, 320])
        cutout(img, alpha)

    def test_is_gray(self) -> None:
        self.assertTrue(is_gray(np.random.random([320, 320, 1])))
        self.assertFalse(is_gray(np.random.random([320, 320, 3])))
        self.assertFalse(is_gray(np.random.random([320, 320, 4])))


if __name__ == "__main__":
    unittest.main()
