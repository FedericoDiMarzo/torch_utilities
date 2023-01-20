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
from torch_utils.modules import get_time_value, get_freq_value
import torch_utils as tu

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

sum_merge = tu.LambdaLayer(lambda x, y: x + y)

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
        self.padding_f = (None, 0, 1, 2)
        self.dilation = (1, 2, 4, (3, 2))
        self.bias = (False, True)
        self.separable = (False, True)
        self.enable_weight_norm = (False, True)
        self.dtype = (torch.float, torch.double)
        self.in_freqs = (32, 33)
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
            self.in_freqs,
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
            in_freqs,
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
            in_freqs,
        ) = p
        x = _get_input(in_channels, in_freqs, dtype)
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
                in_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                self.assertEqual(type(conv.layers), nn.Sequential)
                padding_layer = nn.Identity if padding_f is not None else nn.ConstantPad2d
                if separable:
                    self.assertEqual(type(conv.layers[0]), nn.ConstantPad2d)
                    self.assertEqual(type(conv.layers[1]), nn.Conv2d)
                    self.assertEqual(type(conv.layers[2]), nn.Conv2d)
                    self.assertEqual(type(conv.layers[3]), padding_layer)
                    self.assertTrue(
                        conv.layers[1].kernel_size == kernel_size
                        or conv.layers[1].kernel_size == (kernel_size, kernel_size)
                    )
                    self.assertEqual(conv.layers[2].kernel_size, (1, 1))
                else:
                    self.assertEqual(type(conv.layers[0]), nn.ConstantPad2d)
                    self.assertEqual(type(conv.layers[1]), nn.Conv2d)
                    self.assertEqual(type(conv.layers[2]), padding_layer)
                    self.assertTrue(
                        conv.layers[1].kernel_size == kernel_size
                        or conv.layers[1].kernel_size == (kernel_size, kernel_size)
                    )

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
                in_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                x = self.get_input(p)
                y = conv(x)
                batch_size, _, frames = x.shape[:3]
                dilation_f = _get_f(dilation)
                kernel_f = _get_f(kernel_size)

                if padding_f is None and stride_f != 1:
                    # auto padding for stride != 1
                    padding_f = 0

                if padding_f is None and stride_f == 1:
                    # auto padding for stride == 1
                    out_freqs = in_freqs
                else:
                    # manual padding
                    out_freqs = in_freqs + 2 * padding_f - dilation_f * (kernel_f - 1) - 1
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
        self.residual_merge = (None, sum_merge)
        self.disable_batchnorm = (False, True)
        self.enable_weight_norm = (False, True)
        self.dtype = (torch.float, torch.double)
        self.in_freqs = (32, 33, 64)
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
            self.in_freqs,
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
            in_freqs,
        ) = p

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
            in_freqs,
        ) = p
        x = _get_input(in_channels, in_freqs, dtype)
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
                in_freqs,
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
                in_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                x = self.get_input(p)
                y = conv(x)
                batch_size, _, frames = x.shape[:3]
                dilation_f = _get_f(dilation)
                kernel_f = _get_f(kernel_size)
                out_freqs = in_freqs // stride_f
                flag = ((dilation_f * (kernel_f - 1) + 1) % 2 == 0) or (in_freqs % 2 == 1)
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
        self.in_channels = (1, 2)
        self.out_channels = (3,)
        self.post_conv_kernel_size = (1, 2, (1, 3))
        self.post_conv_count = (1, 2)
        self.post_conv_dilation = (None, 1, 3)
        self.tconv_stride_f = (1, 2)
        self.separable = (False, True)
        self.disable_batchnorm = (False, True)
        self.enable_weight_norm = (False, True)
        self.activation = (None, nn.ReLU())
        self.residual_merge = (None, sum_merge)
        self.dtype = (torch.float, torch.double)
        self.in_freqs = (50, 51)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.params = product(
            self.in_channels,
            self.out_channels,
            self.post_conv_kernel_size,
            self.post_conv_count,
            self.post_conv_dilation,
            self.tconv_stride_f,
            self.separable,
            self.disable_batchnorm,
            self.enable_weight_norm,
            self.activation,
            self.residual_merge,
            self.dtype,
            self.in_freqs,
        )

    def get_instance(self, p: Tuple) -> tu.CausalConvNeuralUpsampler:
        (
            in_channels,
            out_channels,
            post_conv_kernel_size,
            post_conv_count,
            post_conv_dilation,
            tconv_stride_f,
            separable,
            disable_batchnorm,
            enable_weight_norm,
            activation,
            residual_merge,
            dtype,
            in_freqs,
        ) = p

        if post_conv_count > 1 and post_conv_dilation is not None:
            post_conv_kernel_size, post_conv_dilation = [
                [y] * post_conv_count for y in (post_conv_kernel_size, post_conv_dilation)
            ]

        instance = tu.CausalConvNeuralUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            post_conv_kernel_size=post_conv_kernel_size,
            post_conv_count=post_conv_count,
            post_conv_dilation=post_conv_dilation,
            tconv_stride_f=tconv_stride_f,
            separable=separable,
            disable_batchnorm=disable_batchnorm,
            enable_weight_norm=enable_weight_norm,
            activation=activation,
            residual_merge=residual_merge,
            dtype=dtype,
        )
        return instance

    def get_input(self, p: Tuple) -> Tensor:
        (
            in_channels,
            out_channels,
            post_conv_kernel_size,
            post_conv_count,
            post_conv_dilation,
            tconv_stride_f,
            separable,
            disable_batchnorm,
            enable_weight_norm,
            activation,
            residual_merge,
            dtype,
            in_freqs,
        ) = p
        x = _get_input(in_channels, in_freqs, dtype)
        return x

    def test_inner_modules(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                post_conv_kernel_size,
                post_conv_count,
                post_conv_dilation,
                tconv_stride_f,
                separable,
                disable_batchnorm,
                enable_weight_norm,
                activation,
                residual_merge,
                dtype,
                in_freqs,
            ) = p
            with self.subTest(p=p):
                upsamp = self.get_instance(p)

                # tconv
                tconv = upsamp.tconv
                self.assertEqual(tconv.in_channels, in_channels)
                self.assertEqual(tconv.out_channels, out_channels)
                self.assertEqual(tconv.kernel_size, (1, 2 * tconv_stride_f))

                # post conv
                pconv = upsamp.conv
                self.assertEqual(type(pconv), nn.Sequential)
                self.assertEqual(len(pconv), post_conv_count * 2)
                for i, c in enumerate(pconv):
                    i = i // 2
                    if isinstance(c, tu.CausalConv2d) and not separable:
                        self.assertEqual(c.conv.in_channels, out_channels)
                        self.assertEqual(c.conv.out_channels, out_channels)
                        self.assertEqual(c.separable, separable)
                        self.assertEqual(c.enable_weight_norm, enable_weight_norm)
                        self.assertTrue(
                            c.conv.kernel_size == post_conv_kernel_size
                            or c.conv.kernel_size == (post_conv_kernel_size, post_conv_kernel_size)
                        )
                        if post_conv_dilation is None:
                            k_t = get_time_value(c.conv.kernel_size)
                            k_f = get_freq_value(c.conv.kernel_size)
                            expected_dilation = (k_t**i, k_f**i)
                            self.assertEqual(c.conv.dilation, expected_dilation)

                # batchnorm
                if enable_weight_norm:
                    self.assertTrue(upsamp.disable_batchnorm)

                # activation
                act = upsamp.activation
                if activation is None:
                    self.assertEqual(type(act), nn.Identity)
                else:
                    self.assertEqual(type(act), type(activation))

    def test_forward(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                post_conv_kernel_size,
                post_conv_count,
                post_conv_dilation,
                tconv_stride_f,
                separable,
                disable_batchnorm,
                enable_weight_norm,
                activation,
                residual_merge,
                dtype,
                in_freqs,
            ) = p
            with self.subTest(p=p):
                upsamp = self.get_instance(p)
                x = self.get_input(p)
                y = upsamp(x)
                batch_size, _, frames = x.shape[:3]
                out_freqs = tconv_stride_f * in_freqs
                expected_shape = (batch_size, out_channels, frames, out_freqs)
                self.assertEqual(y.shape, expected_shape)


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
