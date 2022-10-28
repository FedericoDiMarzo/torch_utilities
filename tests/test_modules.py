from typing import Callable
import unittest
from pathlib import Path
from pathimport import set_module_root
from torch import Tensor, nn
import numpy as np
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as TU
from tests.generate_test_data import get_test_data_dir
from torch_utils.common import repeat_test

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
    def test_conv(self):
        conv = TU.CausalConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
        )
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_padding(self):
        conv = TU.CausalConv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(0, 1))
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_separable(self):
        conv = TU.CausalConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            separable=True,
        )
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_dilation(self):
        conv = TU.CausalConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            dilation=(12, 1),
        )
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)


class TestCausalConv2dNormAct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    @torch.no_grad()
    def test_conv(self):
        conv = TU.CausalConv2dNormAct(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
        )
        x = torch.ones((1, 1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_separable(self):
        conv = TU.CausalConv2dNormAct(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            separable=True,
        )
        x = torch.ones((1, 1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_sum(self):
        conv = TU.CausalConv2dNormAct(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            residual_merge=lambda x, y: x + y,
        )
        x = torch.ones((1, 1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_concat(self):
        conv = TU.CausalConv2dNormAct(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            residual_merge=lambda x, y: torch.concat([x, y], dim=1),
        )
        x = torch.ones((1, 1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, (1, 2, 100, 3))


class TestReparameterize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.reparam = TU.Reparameterize()
        self.eps = 1e-2

    @repeat_test(5)
    @torch.no_grad()
    def test_zero_mean(self):
        mu = torch.zeros(100000)
        logvar = torch.ones(100000)
        y = self.reparam(mu, logvar)
        y = torch.mean(y)
        self.assertLess(y.item(), self.eps)

    @repeat_test(5)
    def test_nonzero_mean(self):
        mu = torch.ones(100000)
        logvar = torch.ones(100000)
        y = self.reparam(mu, logvar)
        y = torch.mean(y) - 1
        self.assertLess(y.item(), self.eps)


class TestScaleChannels2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.scale = TU.ScaleChannels2d(2)

    def set_weights(self, weights: list):
        w = self.scale.scale.weight.data
        self.scale.scale.weight.data = torch.ones_like(w)
        for i, w in enumerate(weights):
            self.scale.scale.weight.data[i] = w

    @torch.no_grad()
    def test_scaling(self):
        self.set_weights([1, 2])
        x = torch.ones((1, 2, 10, 32))
        y = self.scale(x)
        self.assertTrue(torch.allclose(y[0, 0], torch.ones_like(y[0, 0])))
        self.assertTrue(torch.allclose(y[0, 1], torch.ones_like(y[0, 1]) * 2))


class TestCausalConvNeuralUpsampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    @torch.no_grad()
    def test_forward(self):
        upsam = TU.CausalConvNeuralUpsampler(
            in_channels=1,
            out_channels=1,
            tconv_kernelf_size=1,
            tconv_padding_f=1,
            tconv_output_padding_f=0,
            conv_kernel_size=(1, 1),
            tconv_stride_f=1,
        )
        x = torch.ones((1, 1, 100, 32))
        y = upsam(x)
        self.assertEqual(y.shape, x.shape)

    # @torch.no_grad()
    # def test_forward_padding(self):
    #     conv = TU.CausalConv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(0, 1))
    #     x = torch.ones((1, 100, 3))
    #     y = conv(x)
    #     self.assertEqual(y.shape, x.shape)

    # @torch.no_grad()
    # def test_forwarf_separable(self):
    #     conv = TU.CausalConv2d(
    #         in_channels=1,
    #         out_channels=1,
    #         kernel_size=(5, 1),
    #         separable=True,
    #     )
    #     x = torch.ones((1, 100, 3))
    #     y = conv(x)
    #     self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
