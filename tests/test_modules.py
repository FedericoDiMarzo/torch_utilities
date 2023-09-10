from torch.nn.utils import weight_norm
from typing import Tuple, Type
from itertools import product
from torch import Tensor
from torch import nn
import numpy as np
import unittest
import torch

import torch_utilities as tu
from torch_utilities import repeat_test, set_device
from torch_utilities.modules import (
    get_causal_longformer_mask,
    get_time_value,
    get_freq_value,
    LambdaLayer,
    Lookahead,
    Reparameterize,
    ScaleChannels2d,
    UnfoldSpectrogram,
    FoldSpectrogram,
    ResidualWrap,
    GroupedLinear,
    CausalConv2d,
    CausalSubConv2d,
    CausalConv2dNormAct,
    CausalSmoothedTConv,
    DenseConvBlock,
    GruNormAct,
    SlidingCausalMultiheadAttention,
)

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

sum_merge = LambdaLayer(lambda x, y: x + y)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestLookahead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.x = torch.zeros(1, 2, 10, 16)

    def test_no_maintain_shape(self):
        lookahead = Lookahead(4)
        y = lookahead(self.x)
        self.assertEqual(y.shape, (1, 2, 6, 16))

    def test_maintain_shape(self):
        lookahead = Lookahead(4, maintain_shape=True)
        y = lookahead(self.x)
        self.assertEqual(y.shape, self.x.shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestUnfoldFoldSpectrogram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_channels = 2
        self.in_frames = 20
        self.in_num_blocks = 3
        self.in_freqs = 4

        self.block_size = (2, 4)
        self.stride = (1, 2)
        self.params = product(self.block_size, self.stride)

    def get_instance(self, p: Tuple, unfold: bool) -> CausalConv2d:
        (block_size, stride) = p
        if unfold:
            instance = UnfoldSpectrogram(
                block_size=block_size,
                stride=stride,
            )
        else:
            instance = FoldSpectrogram(
                block_size=block_size,
                stride=stride,
                channels=self.in_channels,
            )
        return instance

    def get_input(self, p: Tuple, unfold: bool) -> Tensor:
        (block_size, stride) = p

        if unfold:
            x = torch.randn((1, self.in_channels, self.in_frames, self.in_freqs))
        else:
            x = torch.randn(
                (1, self.in_channels, self.in_num_blocks, block_size, self.in_freqs)
            )
        return x

    def test_unfold(self):
        for p in self.params:
            (block_size, stride) = p
            with self.subTest(p=p):
                unfold = self.get_instance(p, unfold=True)
                x = self.get_input(p, unfold=True)
                y = unfold(x)

                n_blocks = (self.in_frames - (block_size - 1) - 1) / stride + 1
                expected_shape = (
                    1,
                    self.in_channels,
                    n_blocks,
                    block_size,
                    self.in_freqs,
                )
                self.assertTrue(y.shape, expected_shape)

    def test_unfold_assertion(self):
        x = torch.ones(1, 1, 11, 4)
        unfold = UnfoldSpectrogram(4, 2)
        with self.assertRaises(AssertionError):
            unfold(x)

    def test_fold(self):
        for p in self.params:
            (block_size, stride) = p
            with self.subTest(p=p):
                fold = self.get_instance(p, unfold=False)
                x = self.get_input(p, unfold=False)
                y = fold(x)
                self.assertEqual(y.shape[1], self.in_channels)
                self.assertEqual(y.shape[3], self.in_freqs)

    def test_inversion(self):
        for p in self.params:
            (block_size, stride) = p
            with self.subTest(p=p):
                unfold = self.get_instance(p, unfold=True)
                fold = self.get_instance(p, unfold=False)
                x = self.get_input(p, unfold=True)
                y = unfold(x)
                x_hat = fold(y)
                e = (x - x_hat).abs().max().item()
                self.assertLess(e, 1e-6)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestResidualWrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.x = torch.ones(1, 2, 10, 16)

    def test_single_layer(self):
        layer = ResidualWrap(nn.Identity())
        y = layer(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertLess((self.x * 2 - y).abs().max(), 1e-6)

    def test_multiple_layers(self):
        layer = ResidualWrap(nn.Identity(), nn.Identity())
        y = layer(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertLess((self.x * 2 - y).abs().max(), 1e-6)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class TestCausalConv2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_channels = (1, 2)
        self.out_channels = (1, 3)
        self.kernel_size = (1, 4, (2, 3))
        self.stride_f = (1, 2, 4)
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

    def get_instance(self, p: Tuple) -> CausalConv2d:
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
        instance = CausalConv2d(
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
                padding_layer = (
                    nn.Identity if padding_f is not None else nn.ConstantPad2d
                )
                if separable:
                    self.assertEqual(type(conv.layers[0]), nn.ConstantPad2d)
                    self.assertEqual(type(conv.layers[1]), padding_layer)
                    self.assertEqual(type(conv.layers[2]), nn.Conv2d)
                    self.assertEqual(type(conv.layers[3]), nn.Conv2d)
                    self.assertTrue(
                        conv.layers[2].kernel_size == kernel_size
                        or conv.layers[2].kernel_size == (kernel_size, kernel_size)
                    )
                    self.assertEqual(conv.layers[3].kernel_size, (1, 1))
                else:
                    self.assertEqual(type(conv.layers[0]), nn.ConstantPad2d)
                    self.assertEqual(type(conv.layers[1]), padding_layer)
                    self.assertEqual(type(conv.layers[2]), nn.Conv2d)
                    self.assertTrue(
                        conv.layers[2].kernel_size == kernel_size
                        or conv.layers[2].kernel_size == (kernel_size, kernel_size)
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

            # skipping autopadding when stride_f doesn't divide in_freqs
            if padding_f is None and in_freqs % stride_f != 0:
                continue

            with self.subTest(p=p):
                conv = self.get_instance(p)
                x = self.get_input(p)
                y = conv(x)
                batch_size, _, frames = x.shape[:3]
                dilation_f = _get_f(dilation)
                kernel_f = _get_f(kernel_size)

                if padding_f is None:
                    # auto padding
                    out_freqs = in_freqs // stride_f
                else:
                    # manual padding
                    out_freqs = (
                        in_freqs + 2 * padding_f - dilation_f * (kernel_f - 1) - 1
                    )
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
        self.merge_after_conv = (False, True)
        self.dtype = (torch.float, torch.double)
        self.in_freqs = (32, 64)
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
            self.merge_after_conv,
            self.dtype,
            self.in_freqs,
        )

    def get_instance(self, p: Tuple) -> CausalConv2dNormAct:
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
            merge_after_conv,
            dtype,
            in_freqs,
        ) = p

        if in_freqs % 2 == 1:
            stride_f = 1

        instance = CausalConv2dNormAct(
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
            merge_after_conv=merge_after_conv,
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
            merge_after_conv,
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
                merge_after_conv,
                dtype,
                in_freqs,
            ) = p
            with self.subTest(p=p):
                conv = self.get_instance(p)
                causal_conv_2d = conv.conv
                self.assertEqual(type(causal_conv_2d), CausalConv2d)
                if enable_weight_norm:
                    disable_batchnorm = True
                    self.assertEqual(causal_conv_2d._normalize, weight_norm)
                if activation is None:
                    self.assertEqual(type(conv.activation), nn.Identity)
                batchnorm = conv.batchnorm
                expected_batchnorm = (
                    nn.Identity if disable_batchnorm else nn.BatchNorm2d
                )
                self.assertEqual(type(batchnorm), expected_batchnorm)

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
                merge_after_conv,
                dtype,
                in_freqs,
            ) = p
            if in_freqs % 2 == 1:
                stride_f = 1

            if (not merge_after_conv) and (stride_f != 1):
                # excluding cases where the merge is before the stride
                # and the stride is different than 1
                return

            with self.subTest(p=p):
                conv = self.get_instance(p)
                x = self.get_input(p)
                y = conv(x)
                batch_size, _, frames = x.shape[:3]
                out_freqs = in_freqs // stride_f
                expected_shape = (batch_size, out_channels, frames, out_freqs)
                self.assertEqual(y.shape, expected_shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestReparameterize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.reparam = Reparameterize()
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
        self.scale = ScaleChannels2d(2)

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
class TestCausalSmoothedTConv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_channels = (1,)
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
        self.dtype = (torch.float,)
        self.in_freqs = (50,)
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

    def get_instance(self, p: Tuple) -> CausalSmoothedTConv:
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

        instance = CausalSmoothedTConv(
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
                self.assertEqual(len(pconv), post_conv_count)
                for i, c in enumerate(pconv):
                    if isinstance(c, CausalConv2d) and not separable:
                        self.assertEqual(c.in_channels, out_channels)
                        self.assertEqual(c.out_channels, out_channels)
                        self.assertEqual(c.separable, separable)
                        self.assertEqual(c.enable_weight_norm, enable_weight_norm)
                        self.assertTrue(
                            c.kernel_size == post_conv_kernel_size
                            or c.kernel_size
                            == (post_conv_kernel_size, post_conv_kernel_size)
                        )
                        if post_conv_dilation is None:
                            k_t = get_time_value(c.kernel_size)
                            k_f = get_freq_value(c.kernel_size)
                            expected_dilation = (k_t**i, k_f**i)
                            self.assertEqual(c.dilation, expected_dilation)

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
                    GroupedLinear(p[0], p[1], groups=8)

    def test_no_groups(self):
        x = torch.rand((1, 10, 32))
        gl = GroupedLinear(32, 64, 1)
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
                gru = GruNormAct(
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
                gru = GruNormAct(
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
class TestDenseConvBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.channels = (2, 4)
        self.kernel_size = (3, (3, 5))
        self.dilation = (None, 1, (2, 5))
        self.disable_dilation_f = (False, True)
        self.depth = (1, 4)
        self.final_stride = (1, 2)
        self.disable_layernorm = (False, True)
        self.enable_weight_norm = (False, True)
        self.activation = (None, nn.ReLU())
        self.feature_size = (40,)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.params = product(
            self.channels,
            self.kernel_size,
            self.dilation,
            self.disable_dilation_f,
            self.depth,
            self.final_stride,
            self.disable_layernorm,
            self.enable_weight_norm,
            self.activation,
            self.feature_size,
        )

    def get_instance(self, p: Tuple) -> DenseConvBlock:
        (
            channels,
            kernel_size,
            dilation,
            disable_dilation_f,
            depth,
            final_stride,
            disable_layernorm,
            enable_weight_norm,
            activation,
            feature_size,
        ) = p
        disable_layernorm = disable_layernorm or enable_weight_norm
        instance = DenseConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            feature_size=feature_size,
            dilation=dilation,
            disable_dilation_f=disable_dilation_f,
            depth=depth,
            final_stride=final_stride,
            disable_layernorm=disable_layernorm,
            enable_weight_norm=enable_weight_norm,
            activation=activation,
        )
        return instance

    def get_input(self, p: Tuple) -> Tensor:
        (
            channels,
            kernel_size,
            dilation,
            disable_dilation_f,
            depth,
            final_stride,
            disable_layernorm,
            enable_weight_norm,
            activation,
            feature_size,
        ) = p
        x = _get_input(channels, feature_size, None)
        return x

    def test_inner_modules(self):
        for p in self.params:
            (
                channels,
                kernel_size,
                dilation,
                disable_dilation_f,
                depth,
                final_stride,
                disable_layernorm,
                enable_weight_norm,
                activation,
                feature_size,
            ) = p
            disable_layernorm = disable_layernorm or enable_weight_norm
            with self.subTest(p=p):
                dcb = self.get_instance(p)
                layers = dcb.layers
                for i, seq in enumerate(layers):
                    expected_layernorm = (
                        nn.Identity if disable_layernorm else nn.LayerNorm
                    )
                    expected_layernorm = (
                        nn.Identity if i == (depth - 1) else expected_layernorm
                    )
                    expected_activation = (
                        nn.Identity if activation is None else type(activation)
                    )
                    self.assertEqual(type(seq), nn.Sequential)
                    self.assertEqual(type(seq[0]), CausalConv2dNormAct)
                    self.assertEqual(type(seq[1]), expected_layernorm)
                    self.assertEqual(type(seq[2]), expected_activation)
                    self.assertEqual(seq[0].in_channels, (i + 1) * channels)
                    self.assertEqual(seq[0].out_channels, channels)
                    self.assertEqual(
                        seq[0].stride_f, 1 if i != (depth - 1) else final_stride
                    )
                    self.assertTrue(seq[0].disable_batchnorm)
                    self.assertEqual(seq[0].enable_weight_norm, enable_weight_norm)

    def test_forward(self):
        for p in self.params:
            (
                channels,
                kernel_size,
                dilation,
                disable_dilation_f,
                depth,
                final_stride,
                disable_layernorm,
                enable_weight_norm,
                activation,
                feature_size,
            ) = p
            disable_layernorm = disable_layernorm or enable_weight_norm
            with self.subTest(p=p):
                dcb = self.get_instance(p)
                x = self.get_input(p)
                y = dcb(x)
                B, C, T, F = x.shape
                self.assertEqual(y.shape, (B, C, T, F // final_stride))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestCausalSubConv2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.in_channels = (2,)
        self.out_channels = (3,)
        self.kernel_size = (4, (5, 3))
        self.stride = (1, 2, 4)
        self.dilation = (1, (2, 3))
        self.bias = (False,)
        self.separable = (False, True)
        self.enable_weight_norm = (False, True)
        self.upsampling_dim = ("time",)
        # self.upsampling_dim = ("freq", "time")
        self.in_freqs = (16,)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.params = product(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.bias,
            self.separable,
            self.enable_weight_norm,
            self.upsampling_dim,
            self.in_freqs,
        )

    def get_instance(self, p: Tuple) -> CausalSubConv2d:
        (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            separable,
            enable_weight_norm,
            upsampling_dim,
            in_freqs,
        ) = p
        instance = CausalSubConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            separable=separable,
            upsampling_dim=upsampling_dim,
            enable_weight_norm=enable_weight_norm,
        )
        return instance

    def get_input(self, p: Tuple) -> Tensor:
        (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            separable,
            enable_weight_norm,
            upsampling_dim,
            in_freqs,
        ) = p
        x = _get_input(in_channels, in_freqs, None)
        return x

    def test_inner_modules(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation,
                bias,
                separable,
                enable_weight_norm,
                upsampling_dim,
                in_freqs,
            ) = p
            with self.subTest(p=p):
                csc = self.get_instance(p)
                self.assertEqual(len(csc.layers), stride)
                for layer in csc.layers:
                    self.assertEqual(layer.in_channels, in_channels)
                    self.assertEqual(layer.out_channels, out_channels)
                    self.assertEqual(layer.kernel_size, kernel_size)
                    self.assertEqual(layer.stride_f, 1)
                    self.assertEqual(layer.dilation, dilation)
                    self.assertEqual(layer.bias, bias)
                    self.assertEqual(layer.separable, separable)
                    self.assertEqual(layer.enable_weight_norm, enable_weight_norm)

    def test_forward(self):
        for p in self.params:
            (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation,
                bias,
                separable,
                enable_weight_norm,
                upsampling_dim,
                in_freqs,
            ) = p
            with self.subTest(p=p):
                csc = self.get_instance(p)
                x = self.get_input(p)
                y = csc(x)
                B, C, T, F = x.shape
                expected_shape = (
                    (B, out_channels, T, in_freqs * stride)
                    if upsampling_dim == "freq"
                    else (B, out_channels, T * stride, in_freqs)
                )
                self.assertEqual(y.shape, expected_shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestSlidingCausalMultiheadAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.channels = (2,)
        self.sequence_len = (10,)
        self.embed_dim = (32, 64)
        self.stride = (5, 2)
        self.num_heads = (1,)
        self.dropout = (0,)
        self.bias = (True,)
        self.receptive_field = (None,)
        self.attn_mask = (None,)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.params = product(
            self.channels,
            self.sequence_len,
            self.embed_dim,
            self.stride,
            self.num_heads,
            self.dropout,
            self.bias,
            self.receptive_field,
            self.attn_mask,
        )

    def get_instance(self, p: Tuple) -> SlidingCausalMultiheadAttention:
        (
            channels,
            sequence_len,
            embed_dim,
            stride,
            num_heads,
            dropout,
            bias,
            receptive_field,
            attn_mask,
        ) = p
        instance = SlidingCausalMultiheadAttention(
            channels=channels,
            sequence_len=sequence_len,
            embed_dim=embed_dim,
            stride=stride,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            receptive_field=receptive_field,
            attn_mask=attn_mask,
        )
        return instance

    def get_input(self, p: Tuple) -> Tensor:
        (
            channels,
            sequence_len,
            embed_dim,
            stride,
            num_heads,
            dropout,
            bias,
            receptive_field,
            attn_mask,
        ) = p
        x = _get_input(channels, embed_dim, None)
        return x

    def test_inner_modules(self):
        for p in self.params:
            (
                channels,
                sequence_len,
                embed_dim,
                stride,
                num_heads,
                dropout,
                bias,
                receptive_field,
                attn_mask,
            ) = p
            with self.subTest(p=p):
                att = self.get_instance(p)
                self.assertEqual(type(att.mh_attention), nn.MultiheadAttention)
                self.assertEqual(type(att.unfold), UnfoldSpectrogram)
                self.assertEqual(type(att.fold), FoldSpectrogram)

                # default attention mask
                if attn_mask is None and receptive_field is None:
                    expected = get_causal_longformer_mask(
                        sequence_len, sequence_len // 2
                    )
                    max_err = (expected - att.attn_mask).abs().max()
                    self.assertLess(max_err, 1e-12)

    def test_forward(self):
        for p in self.params:
            (
                channels,
                sequence_len,
                embed_dim,
                stride,
                num_heads,
                dropout,
                bias,
                receptive_field,
                attn_mask,
            ) = p
            with self.subTest(p=p):
                att = self.get_instance(p)
                x = self.get_input(p)
                y = att(x)
                self.assertEqual(x.shape, y.shape)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

if __name__ == "__main__":
    unittest.main(verbosity=2)
