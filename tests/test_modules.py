from pathimport import set_module_root
from typing import Callable
from pathlib import Path
from torch import Tensor, nn
import numpy as np
import itertools
import unittest
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as tu
from tests.generate_test_data import get_test_data_dir
from torch_utils.common import repeat_test, to_numpy

torch.manual_seed(984)
np.random.seed(876)
D = tu.get_device()


class TestLookahead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.x = torch.zeros(1, 2, 10, 16, device=D)

    @torch.no_grad()
    def test_no_maintain_shape(self):
        lookahead = tu.Lookahead(4)
        y = lookahead(self.x)
        self.assertEqual(y.shape, (1, 2, 6, 16))

    @torch.no_grad()
    def test_maintain_shape(self):
        lookahead = tu.Lookahead(4, maintain_shape=True)
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
        conv = tu.CausalConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            device=D,
        )
        x = torch.ones((1, 100, 3), device=D)
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_padding(self):
        conv = tu.CausalConv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding_f=1)
        x = torch.ones((1, 100, 3))
        y = conv(x)
        self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_separable(self):
        conv = tu.CausalConv2d(
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
        conv = tu.CausalConv2d(
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
        params = (
            (1, 4, 5),  # kernel_f
            (1, 2, 5),  # kernel_t
            (32, 33),  # freq_bins
            (False, True),  # separable
        )
        params = itertools.product(*params)
        for p in params:
            with self.subTest(p=p):
                conv = tu.CausalConv2dNormAct(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=(p[0], p[1]),
                    separable=p[3],
                )
                x = torch.ones((1, 1, 100, p[2]))
                y = conv(x)
                self.assertEqual(y.shape, x.shape)

    @torch.no_grad()
    def test_conv_sum(self):
        conv = tu.CausalConv2dNormAct(
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
        conv = tu.CausalConv2dNormAct(
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
        self.reparam = tu.Reparameterize()
        self.eps = 1e-1

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
        self.scale = tu.ScaleChannels2d(2)

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
    def test_forward1(self):
        params = (
            (1, 3, 4, 21),  # tconv_kernelf_size
            (1, 2, 4),  # stride
            (1, 3),  # conv_kernel_size
            (False, True),  # separable
            (False, True),  # with sum?
        )
        params = itertools.product(*params)
        for p in params:
            # residual_merge
            if p[4] and p[1] == 1:
                merge = lambda x, y: x + y
            else:
                merge = None

            with self.subTest(p=p):
                upsam = tu.CausalConvNeuralUpsampler(
                    in_channels=1,
                    out_channels=1,
                    tconv_kernel_f_size=p[0],
                    tconv_padding_f=0,
                    conv_kernel_size=(1, p[2]),
                    tconv_stride_f=p[1],
                    separable=p[3],
                    residual_merge=merge,
                )
                x = torch.ones((1, 1, 100, 32))
                y = upsam(x)
                self.assertEqual(list(y.shape), [*x.shape[:-1], 32 * p[1]])


class TestGroupedLinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_assert(self):
        params = (
            (65, 32),
            (64, 31),
        )
        for p in params:
            with self.subTest(p=p):
                with self.assertRaises(AssertionError):
                    tu.GroupedLinear(p[0], p[1], groups=8)

    def test_no_groups(self):
        x = torch.rand((1, 10, 32)).to(tu.get_device())
        gl = tu.GroupedLinear(32, 64, 1)
        lin = nn.Linear(32, 64, bias=False).to(tu.get_device())
        gl.weight.data = torch.ones_like(gl.weight.data)
        lin.weight.data = torch.ones_like(lin.weight.data)
        self.assertTrue(torch.allclose(gl(x), lin(x)))


if __name__ == "__main__":
    unittest.main()
