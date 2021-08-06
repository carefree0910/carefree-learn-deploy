import unittest

import numpy as np

from cflearn_deploy.toolkit import cutout


class TestToolkit(unittest.TestCase):
    def test_cutout(self) -> None:
        img = np.random.random([320, 320, 3])
        alpha = np.random.random([320, 320])
        cutout(img, alpha)


if __name__ == "__main__":
    unittest.main()
