from itertools import product
import unittest
from pathimport import set_module_root
from torch import Tensor
import numpy as np
import torch

set_module_root("../torch_utils")
from tests.generate_test_data import get_test_data_dir
from torch_utilities.metrics import DNSMOS
import torch_utilities as tu


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        pass


class TestDNSMOS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.dnsmos = DNSMOS()

    def test_compute_features(self):
        module = (np, torch)
        for m in module:
            with self.subTest(m=m):
                B, C, T = (2, 1, 16000)
                x = m.zeros((B, C, T))
                y = self.dnsmos.compute_features(x)
                self.assertEqual(y.shape[:2], (B, C))


if __name__ == "__main__":
    unittest.main(verbosity=2)
