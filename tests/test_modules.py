from typing import Tuple, Type
from pathimport import set_module_root
from torch.nn.utils import weight_norm
from itertools import product
from torch import Tensor
from torch import nn
import numpy as np
import unittest
import torch

set_module_root("../torch_utils")
from torch_utils import repeat_test, set_device
import torch_utils as tu

# TODO: rewrite modules tests

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(876)
    set_device("auto")
    torch.set_grad_enabled(False)


def _get_input(in_channels: int, in_freqs: int, dtype: Type) -> Tuple:
    batch_size = 1
    frames = 30
    x = torch.randn(batch_size, in_channels, frames, in_freqs, dtype=dtype)
    return x


_get_f = lambda x: x if isinstance(x, int) else x[1]

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestLookahead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.x = torch.zeros(1, 2, 10, 16)

    def test_no_maintain_shape(self):
        lookahead = tu.Lookahead(4)
        y = lookahead(self.x)
        self.assertEqual(y.shape, (1, 2, 6, 16))

    def test_maintain_shape(self):
        lookahead = tu.Lookahead(4, maintain_shape=True)
        y = lookahead(self.x)
        self.assertEqual(y.shape, self.x.shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestCausalConv2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_channels = (1, 2)
        self.out_channels = (1, 3)
        self.kernel_size = (1, 4, (2, 3))
        self.stride_f = (1, 2, 3, 4)
        self.padding_f = (0, 1, 2)
        self.dilation = (1, 2, 4, (3, 2))
        self.bias = (False, True)
        self.separable = (False, True)
        self.enable_weight_norm = (False, True)
        self.dtype = (torch.float, torch.double)
        self.input_freqs = (32, 33)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.params = product(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride_f,
            self.padding_f,
            self.dilation,
            self.bias,
            self.separable,
            self.enable_weight_norm,
            self.dtype,
            self.input_freqs,
        )

    def get_instance(self, p: Tuple) -> tu.CausalConv2d:
        (
            in_channels,
            out_channels,
            kernel_size,
            stride_f,
            padding_f,
            dilation,
            bias,
            separable,
            enable_weight_norm,
            dtype,
            input_freqs,
        ) = p
        instance = tu.CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride_f=stride_f,
            padding_f=padding_f,
            dilation=dilation,
            bias=bias,
            separable=separable,
            enable_weight_norm=enable_weight_norm,
            dtype=dtype,
        )
        return instance

    def get_input(self, p: Tuple) -> Tensor:
        (
            in_channels,
            out_channels,
            kernel_size,
            stride_f,
            padding_f,
            dilation,
            bias,
            separable,
            enable_weight_norm,
            dtype,
            input_freqs,
        ) = p
        x = _get_input(in_channels, input_freqs, dtype)
        return x

    def test_inner_modules(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                kernel_size,
                stride_f,
                padding_f,
                dilation,
                bias,
                separable,
                enable_weight_norm,
                dtype,
                input_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                if separable:
                    self.assertEqual(type(conv.conv), nn.Sequential)
                    self.assertEqual(type(conv.conv[0]), nn.Conv2d)
                    self.assertEqual(type(conv.conv[1]), nn.Conv2d)
                else:
                    self.assertEqual(type(conv.conv), nn.Conv2d)

                if enable_weight_norm:
                    self.assertEqual(conv._normalize, weight_norm)

    def test_forward(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                kernel_size,
                stride_f,
                padding_f,
                dilation,
                bias,
                separable,
                enable_weight_norm,
                dtype,
                input_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                x = self.get_input(p)
                y = conv(x)
                batch_size, _, frames = x.shape[:3]
                dilation_f = _get_f(dilation)
                kernel_f = _get_f(kernel_size)
                out_freqs = input_freqs + 2 * padding_f - dilation_f * (kernel_f - 1) - 1
                out_freqs = int(out_freqs / stride_f + 1)
                expected_shape = (batch_size, out_channels, frames, out_freqs)
                self.assertEqual(y.shape, expected_shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestCausalConv2dNormAct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_channels = (1, 2)
        self.out_channels = (1, 3)
        self.kernel_size = (1, 4, (2, 3))
        self.stride_f = (2, 4)
        self.dilation = (1, 4, (3, 2))
        self.separable = (False, True)
        self.activation = (None, nn.ReLU())
        self.residual_merge = (None, nn.Identity())
        self.disable_batchnorm = (False, True)
        self.enable_weight_norm = (False, True)
        self.dtype = (torch.float, torch.double)
        self.input_freqs = (32, 33, 64)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.params = product(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride_f,
            self.dilation,
            self.separable,
            self.activation,
            self.residual_merge,
            self.disable_batchnorm,
            self.enable_weight_norm,
            self.dtype,
            self.input_freqs,
        )

    def get_instance(self, p: Tuple) -> tu.CausalConv2dNormAct:
        (
            in_channels,
            out_channels,
            kernel_size,
            stride_f,
            dilation,
            separable,
            activation,
            residual_merge,
            disable_batchnorm,
            enable_weight_norm,
            dtype,
            input_freqs,
        ) = p
        if stride_f != 1:
            residual_merge = None

        instance = tu.CausalConv2dNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride_f=stride_f,
            dilation=dilation,
            separable=separable,
            activation=activation,
            residual_merge=residual_merge,
            disable_batchnorm=disable_batchnorm,
            enable_weight_norm=enable_weight_norm,
            dtype=dtype,
        )
        return instance

    def get_input(self, p: Tuple) -> Tensor:
        (
            in_channels,
            out_channels,
            kernel_size,
            stride_f,
            dilation,
            separable,
            activation,
            residual_merge,
            disable_batchnorm,
            enable_weight_norm,
            dtype,
            input_freqs,
        ) = p
        x = _get_input(in_channels, input_freqs, dtype)
        return x

    def test_inner_modules(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                kernel_size,
                stride_f,
                dilation,
                separable,
                activation,
                residual_merge,
                disable_batchnorm,
                enable_weight_norm,
                dtype,
                input_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                causal_conv_2d = conv.conv
                if separable:
                    self.assertEqual(type(causal_conv_2d.conv), nn.Sequential)
                    self.assertEqual(type(causal_conv_2d.conv[0]), nn.Conv2d)
                    self.assertEqual(type(causal_conv_2d.conv[1]), nn.Conv2d)
                else:
                    self.assertEqual(type(causal_conv_2d.conv), nn.Conv2d)

                if enable_weight_norm:
                    self.assertEqual(causal_conv_2d._normalize, weight_norm)

                if activation is None:
                    self.assertEqual(type(conv.activation), nn.Identity)

    def test_forward(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                kernel_size,
                stride_f,
                dilation,
                separable,
                activation,
                residual_merge,
                disable_batchnorm,
                enable_weight_norm,
                dtype,
                input_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                x = self.get_input(p)
                y = conv(x)
                batch_size, _, frames = x.shape[:3]
                dilation_f = _get_f(dilation)
                kernel_f = _get_f(kernel_size)
                out_freqs = input_freqs // stride_f
                flag = ((dilation_f * (kernel_f - 1) + 1) % 2 == 0) or (input_freqs % 2 == 1)
                out_freqs = (out_freqs + 1) if flag else out_freqs
                expected_shape = (batch_size, out_channels, frames, out_freqs)
                self.assertEqual(y.shape, expected_shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestReparameterize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.reparam = tu.Reparameterize()
        self.eps = 1e-1

    @repeat_test(5)
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


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestScaleChannels2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.scale = tu.ScaleChannels2d(2)

    def set_weights(self, weights: list):
        w = self.scale.scale.weight.data
        self.scale.scale.weight.data = torch.ones_like(w)
        for i, w in enumerate(weights):
            self.scale.scale.weight.data[i] = w

    def test_scaling(self):
        self.set_weights([1, 2])
        x = torch.ones((1, 2, 10, 32))
        y = self.scale(x)
        self.assertTrue(torch.allclose(y[0, 0], torch.ones_like(y[0, 0])))
        self.assertTrue(torch.allclose(y[0, 1], torch.ones_like(y[0, 1]) * 2))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestCausalConvNeuralUpsampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        pass

    def test_forward1(self):
        params = (
            (1, 3, 4, 21),  # tconv_kernelf_size
            (1, 2, 4),  # stride
            (1, 3),  # conv_kernel_size
            (False, True),  # separable
            (False, True),  # with sum?
        )
        params = product(*params)
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
                    post_conv_kernel_size=(1, p[2]),
                    tconv_stride_f=p[1],
                    separable=p[3],
                    residual_merge=merge,
                )
                x = torch.ones((1, 1, 100, 32))
                y = upsam(x)
                self.assertEqual(list(y.shape), [*x.shape[:-1], 32 * p[1]])


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestGroupedLinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

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
        x = torch.rand((1, 10, 32))
        gl = tu.GroupedLinear(32, 64, 1)
        lin = nn.Linear(32, 64, bias=False)
        gl.weight.data = torch.ones_like(gl.weight.data)
        lin.weight.data = torch.ones_like(lin.weight.data)
        self.assertTrue(torch.allclose(gl(x), lin(x)))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestMergeLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_ch = (1, 2, 3)
        self.out_ch = (6, 12)
        self.strides = (1, 2, 4)

    def test_down_merge(self):
        params = product(
            self.in_ch,
            self.out_ch,
            self.strides,
        )
        for i, o, s in params:
            with self.subTest(i=i, o=o, s=s):
                merge = tu.DownMerge(o, s)
                x = torch.ones(1, i, 10, 4 * s)
                y = torch.ones(1, o, 10, 4)
                z = merge(x, y)
                self.assertEqual(y.shape, z.shape)

    def test_up_merge(self):
        params = product(
            self.out_ch,
            self.in_ch,
            self.strides,
        )
        for i, o, s in params:
            with self.subTest(i=i, o=o, s=s):
                merge = tu.UpMerge(o, s)
                x = torch.ones(1, i, 10, 4)
                y = torch.ones(1, o, 10, 4 * s)
                z = merge(x, y)
                self.assertEqual(y.shape, z.shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestGruNormAct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_size = (16, 32)
        self.h_size = (8, 64)
        self.merge = (None, (lambda x, y: y))
        self.activation = (None, nn.Sigmoid())
        self.batchnorm = (True, False)
        self.params = product(
            self.in_size,
            self.h_size,
            self.merge,
            self.activation,
            self.batchnorm,
        )

    def get_input(self, in_size: int) -> Tensor:
        x = torch.randn((1, 10, in_size))
        return x

    def test_submodules(self):
        for p in self.params:
            in_size, h_size, merge, act, batchnorm = p
            with self.subTest(p=p):
                gru = tu.GruNormAct(
                    input_size=in_size,
                    hidden_size=h_size,
                    residual_merge=merge,
                    activation=act,
                    disable_batchnorm=not batchnorm,
                )
                submodules = set(type(m) for m in tu.get_submodules(gru))
                batchnorm_type = nn.BatchNorm1d if batchnorm else nn.Identity
                act_type = nn.Identity if act is None else type(act)
                expected = set((nn.GRU, batchnorm_type, act_type))
                self.assertEqual(expected, submodules)

    def test_forward(self):
        for p in self.params:
            in_size, h_size, merge, act, batchnorm = p
            with self.subTest(p=p):
                x = self.get_input(in_size)
                gru = tu.GruNormAct(
                    input_size=in_size,
                    hidden_size=h_size,
                    residual_merge=merge,
                    activation=act,
                    disable_batchnorm=not batchnorm,
                )
                y, h = gru(x)
                self.assertEqual(y.shape, (*x.shape[:-1], h_size))
                self.assertEqual(h.shape, (x.shape[0], 1, h_size))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

if __name__ == "__main__":
    unittest.main(verbosity=2)
