from itertools import product
import onnxruntime as ort
import numpy as np
import unittest
import torch

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

    def test_get_inference_session(self):
        sess = self.dnsmos.get_inference_session()
        self.assertEqual(type(sess), ort.InferenceSession)

    def test_call(self):
        module = (np, torch)
        channels = (1, 2)
        length = (320000,)
        for p in product(module, channels, length):
            m, c, t = p
            with self.subTest(p=p):
                x = m.zeros((c, t))
                scores = self.dnsmos(x)
                self.assertEqual(len(scores), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
