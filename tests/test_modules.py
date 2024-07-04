from typing import Callable, Optional
from torch import nn, Tensor
from torch.nn import Module
import torch
import pytest


from torch_utilities.utilities import OneOrPair

from torch_utilities.modules import (
    SlidingCausalMultiheadAttention,
    CausalConv2dNormAct,
    UnfoldSpectrogram,
    ScaleChannels2d,
    FoldSpectrogram,
    CausalSubConv2d,
    get_freq_value,
    Reparameterize,
    DenseConvBlock,
    GroupedLinear,
    ResidualWrap,
    CausalConv2d,
    GruNormAct,
    LookAhead,
)


# Local fixtures ===============================================================


@pytest.fixture(params=[2])
def channels(request) -> int:
    """Number of channels."""
    return request.param


@pytest.fixture(params=[True, False])
def lookahead_maintain_shape(request) -> bool:
    """Maintain shape in Lookahead."""
    return request.param


# unfold block sie
@pytest.fixture(params=[2, 4])
def unfold_block_size(request) -> int:
    """Unfold Block size."""
    return request.param


@pytest.fixture(params=[1, 2])
def unfold_stride(request) -> int:
    """Unfold stride."""
    return request.param


@pytest.fixture
def unfold_input(channels) -> Tensor:
    """Unfold input."""
    return torch.randn(1, channels, 10, 16)


@pytest.fixture
def fold_input(unfold_block_size, channels) -> Tensor:
    """Unfold output."""
    return torch.randn(1, channels, 3, unfold_block_size, 16)


@pytest.fixture
def unfold_instance(unfold_block_size, unfold_stride) -> UnfoldSpectrogram:
    """Unfold instance."""
    return UnfoldSpectrogram(unfold_block_size, unfold_stride)


@pytest.fixture
def fold_instance(unfold_block_size, unfold_stride, channels) -> FoldSpectrogram:
    """Fold instance."""
    return FoldSpectrogram(unfold_block_size, unfold_stride, channels)


@pytest.fixture(params=[4, (2, 3)])
def kernel_size(request) -> OneOrPair:
    """Conv kernel size."""
    return request.param


@pytest.fixture(params=[1, 4])
def stride_f(request) -> int:
    """Conv stride over frequencies."""
    return request.param


@pytest.fixture(params=[None, 0, 2])
def padding_f(request) -> Optional[int]:
    """Conv padding over frequencies."""
    return request.param


@pytest.fixture(params=[1, (3, 2)])
def dilation(request) -> OneOrPair:
    """Conv dilation."""
    return request.param


@pytest.fixture(params=[True])
def bias(request) -> bool:
    """Conv bias."""
    return request.param


@pytest.fixture(params=[True, False])
def separable(request) -> bool:
    """Enable separable convolution."""
    return request.param


@pytest.fixture(params=[2])
def in_channels(request) -> int:
    """Layer input channels."""
    return request.param


@pytest.fixture(params=[3])
def out_channels(request) -> int:
    """Layer output channels."""
    return request.param


@pytest.fixture(params=[10])
def n_frames(request) -> int:
    """Number of STFT frames."""
    return request.param


@pytest.fixture(params=[16])
def in_freqs(request) -> int:
    """Number of input frequencies."""
    return request.param


@pytest.fixture(params=[1])
def batch_size(request) -> int:
    """Batch size."""
    return request.param


@pytest.fixture()
def stft_input(in_channels, batch_size, n_frames, in_freqs) -> Tensor:
    """STFT input tensor."""
    return torch.randn(batch_size, in_channels, n_frames, in_freqs)


@pytest.fixture
def causal_conv2d_instance(
    in_channels,
    out_channels,
    kernel_size,
    stride_f,
    padding_f,
    dilation,
    bias,
    separable,
) -> CausalConv2d:
    """CausalConv2d instance."""
    return CausalConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride_f=stride_f,
        padding_f=padding_f,
        dilation=dilation,
        bias=bias,
        separable=separable,
    )


@pytest.fixture
def causal_conv2d_norm_act_instance(
    in_channels,
    out_channels,
    kernel_size,
    stride_f,
    dilation,
    separable,
) -> CausalConv2dNormAct:
    """CausalConv2dNormAct instance."""
    return CausalConv2dNormAct(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride_f=stride_f,
        dilation=dilation,
        separable=separable,
    )


@pytest.fixture(params=[8])
def hidden_size(request) -> int:
    """Hidden size."""
    return request.param


@pytest.fixture(params=[None, (lambda x, y: y)])
def merge(request) -> Optional[Callable]:
    """Merge operation."""
    return request.param


@pytest.fixture(params=[None, nn.Sigmoid()])
def activation(request) -> Optional[Module]:
    """Activation function."""
    return request.param


@pytest.fixture(params=[True, False])
def enable_batchnorm(request) -> bool:
    """Enable batchnorm."""
    return request.param


@pytest.fixture
def gru_norm_act_instance(
    in_freqs,
    hidden_size,
    merge,
    activation,
    enable_batchnorm,
) -> GruNormAct:
    """GruNormAct instance."""
    return GruNormAct(
        input_size=in_freqs,
        hidden_size=hidden_size,
        residual_merge=merge,
        activation=activation,
        disable_batchnorm=not enable_batchnorm,
    )


@pytest.fixture
def gru_input(in_freqs) -> Tensor:
    """GRU input."""
    return torch.randn(1, 10, in_freqs)


@pytest.fixture(params=[1, 2])
def final_stride(request) -> int:
    """Final stride for DenseConvBlock."""
    return request.param


@pytest.fixture(params=[1, 4])
def depth(request) -> int:
    """Depth of DenseConvBlock."""
    return request.param


@pytest.fixture(params=[False, True])
def disable_dilation_f(request) -> bool:
    """Disable dilation."""
    return request.param


@pytest.fixture
def dense_conv_block_instance(
    channels,
    kernel_size,
    dilation,
    disable_dilation_f,
    depth,
    stride_f,
    activation,
    in_freqs,
    enable_batchnorm,
) -> DenseConvBlock:
    """DenseConvBlock instance."""
    return DenseConvBlock(
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        disable_dilation_f=disable_dilation_f,
        depth=depth,
        final_stride=stride_f,
        disable_layernorm=not enable_batchnorm,
        activation=activation,
        feature_size=in_freqs,
    )


@pytest.fixture
def causal_sub_conv2d_instance(
    in_channels,
    out_channels,
    kernel_size,
    stride_f,
    dilation,
    separable,
) -> CausalSubConv2d:
    return CausalSubConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride_f,
        dilation=dilation,
        bias=False,
        separable=separable,
    )


@pytest.fixture
def sliding_causal_multihead_attention_instance(
    stride_f, in_freqs
) -> SlidingCausalMultiheadAttention:
    return SlidingCausalMultiheadAttention(
        channels=2,
        sequence_len=10,
        embed_dim=in_freqs,
        stride=stride_f,
        num_heads=2,
        dropout=0,
        bias=False,
        receptive_field=None,
        attn_mask=None,
    )


# ==============================================================================


class TestLookAhead:
    x = torch.zeros(1, 2, 10, 16)

    def test_forward_shape(self, lookahead_maintain_shape):
        expected_shape = (1, 2, 6, 16) if lookahead_maintain_shape else (1, 2, 6, 16)
        lookahead = LookAhead(4)
        y = lookahead(self.x)
        assert y.shape == expected_shape


class TestUnfoldFoldSpectrogram:
    def test_unfold_forward_shape(
        self, unfold_instance, unfold_input, unfold_block_size, unfold_stride
    ):
        batch_size, channels, frames, freqs = unfold_input.shape
        unfold = unfold_instance
        x = unfold_input
        y = unfold(x)

        n_blocks = (frames - (unfold_block_size - 1) - 1) / unfold_stride + 1
        expected_shape = (
            batch_size,
            channels,
            n_blocks,
            unfold_block_size,
            freqs,
        )
        assert y.shape == expected_shape

    def test_fold_forward_shape(self, fold_instance, fold_input):
        batch_size, channels, blocks, block_size, freqs = fold_input.shape
        fold = fold_instance
        x = fold_input
        y = fold(x)
        assert y.shape[:2] == (batch_size, channels)
        assert y.shape[3] == freqs

    def test_inversion(self, fold_instance, unfold_instance, unfold_input):
        unfold = unfold_instance
        fold = fold_instance
        x = unfold_input
        y = unfold(x)
        x_hat = fold(y)
        torch.testing.assert_close(x, x_hat)


class TestResidualWrap:
    x = torch.ones(1, 2, 10, 16)

    def test_single_layer(self):
        layer = ResidualWrap(nn.Identity())
        y = layer(self.x)
        assert y.shape == self.x.shape
        torch.testing.assert_close(2 * self.x, y)

    def test_multiple_layers(self):
        layer = ResidualWrap(nn.Identity(), nn.Identity())
        y = layer(self.x)
        assert y.shape == self.x.shape
        torch.testing.assert_close(2 * self.x, y)


class TestCausalConv2d:
    def test_forward_shape(
        self,
        causal_conv2d_instance,
        padding_f,
        stride_f,
        in_freqs,
        stft_input,
        dilation,
        batch_size,
        out_channels,
        kernel_size,
        n_frames,
    ):
        if padding_f is None and in_freqs % stride_f != 0:
            pytest.skip(
                "Skipping autopadding test when stride_f doesn't divide in_freqs"
            )

        conv = causal_conv2d_instance
        x = stft_input
        y = conv(x)

        dilation_f = get_freq_value(dilation)
        kernel_f = get_freq_value(kernel_size)

        if padding_f is None:
            # auto padding
            out_freqs = in_freqs // stride_f
        else:
            # manual padding
            out_freqs = in_freqs + 2 * padding_f - dilation_f * (kernel_f - 1) - 1
            out_freqs = int(out_freqs / stride_f + 1)

        expected_shape = (batch_size, out_channels, n_frames, out_freqs)
        assert y.shape == expected_shape


class TestCausalConv2dNormAct:
    def test_forward_shape(
        self,
        causal_conv2d_norm_act_instance,
        stride_f,
        in_freqs,
        stft_input,
        batch_size,
        out_channels,
        n_frames,
    ):
        conv = causal_conv2d_norm_act_instance
        x = stft_input
        y = conv(x)
        out_freqs = in_freqs // stride_f
        expected_shape = (batch_size, out_channels, n_frames, out_freqs)
        assert y.shape == expected_shape


class TestReparameterize:
    reparam = Reparameterize()
    eps = 1e-1

    def test_zero_mean(self):
        mu = torch.zeros(100000)
        logvar = torch.ones(100000)
        y = self.reparam(mu, logvar)
        y = torch.mean(y)
        assert y.item() < self.eps

    def test_nonzero_mean(self):
        mu = torch.ones(100000)
        logvar = torch.ones(100000)
        y = self.reparam(mu, logvar)
        y = torch.mean(y) - 1
        assert y.item() < self.eps


class TestScaleChannels2d:
    scale = ScaleChannels2d(2)

    def set_weights(self, weights: list):
        w = self.scale.scale.weight.data
        self.scale.scale.weight.data = torch.ones_like(w)
        for i, w in enumerate(weights):
            self.scale.scale.weight.data[i] = w

    def test_scaling(self):
        self.set_weights([1, 2])
        x = torch.ones((1, 2, 10, 32))
        y = self.scale(x)
        torch.testing.assert_close(y[0, 0], x[0, 0])
        torch.testing.assert_close(y[0, 1], x[0, 1] * 2)


class TestGroupedLinear:
    def test_assert(self):
        with pytest.raises(AssertionError):
            GroupedLinear(64, 11, groups=8)

    def test_no_groups(self):
        x = torch.rand((1, 10, 32))
        gl = GroupedLinear(32, 64, 1)
        lin = nn.Linear(32, 64, bias=False)
        gl.weight.data = torch.ones_like(gl.weight.data)
        lin.weight.data = torch.ones_like(lin.weight.data)
        torch.testing.assert_close(gl(x), lin(x))


class TestGruNormAct:
    def test_forward_shape(self, gru_norm_act_instance, gru_input, hidden_size):
        x = gru_input
        gru = gru_norm_act_instance
        y, h = gru(x)
        assert y.shape == (*x.shape[:-1], hidden_size)
        assert h.shape == (x.shape[0], 1, hidden_size)


class TestDenseConvBlock:
    def test_forward_shape(self, stft_input, dense_conv_block_instance, stride_f):
        dcb = dense_conv_block_instance
        x = stft_input
        y = dcb(x)
        B, C, T, F = x.shape
        assert y.shape == (B, C, T, F // stride_f)


class TestCausalSubConv2d:
    def test_forward(
        self,
        causal_sub_conv2d_instance,
        stft_input,
        out_channels,
        stride_f,
    ):
        csc = causal_sub_conv2d_instance
        x = stft_input
        y = csc(x)
        B, C, T, F = x.shape
        expected_shape = (B, out_channels, T, F * stride_f)
        assert y.shape == expected_shape


class TestSlidingCausalMultiheadAttention:
    def test_forward_shape(
        self, sliding_causal_multihead_attention_instance, stft_input
    ):
        att = sliding_causal_multihead_attention_instance
        x = stft_input
        y = att(x)
        assert x.shape == y.shape
