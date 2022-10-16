from typing import Callable
import unittest
from pathlib import Path
from pathimport import set_module_root
from torch import Tensor
import numpy as np
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as TU
from tests.generate_test_data import get_test_data_dir

torch.manual_seed(984)
np.random.seed(876)


class TestLookahead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.x = torch.zeros(1, 2, 10, 16)

    @torch.no_grad()
    def test_no_maintain_shape(self):
        lookahead = TU.Lookahead(4)
        y = lookahead(self.x)
        self.assertEqual(y.shape, (1, 2, 6, 16))

    @torch.no_grad()
    def test_maintain_shape(self):
        lookahead = TU.Lookahead(4, maintain_shape=True)
        y = lookahead(self.x)
        self.assertEqual(y.shape, self.x.shape)


class TestCausalConv2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    @torch.no_grad()
    def test_conv_padding(self):
        conv = TU.CausalConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
        )
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_padding_dilation(self):
        conv = TU.CausalConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            dilation=(12, 1),
        )
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
