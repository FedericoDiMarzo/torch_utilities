__all__ = [
    # utilities
    "LambdaLayer",
    "LookAhead",
    "Reparameterize",
    "ScaleChannels2d",
    "UnfoldSpectrogram",
    "FoldSpectrogram",
    "ResidualWrap",
    # dense variants
    "GroupedLinear",
    # conv2d variants
    "CausalConv2d",
    "CausalSubConv2d",
    # conv2d compositions
    "CausalConv2dNormAct",
    "DenseConvBlock",
    # recurrent variants
    "GruNormAct",
    # attention variants
    "SlidingCausalMultiheadAttention",
]

from typing import Callable, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
import numpy as np
import torch


from torch_utilities.audio import interleave
from torch_utilities.utilities import OneOrPair


# private utility functions = = = = = = = = = = = = = =


def get_time_value(param):
    """
    Extracts the parameter referring to the
    temporal axis.

    Parameters
    ----------
    param : tuple or scalar
        Module parameter

    Returns
    -------
    scalar
        Temporal parameter
    """
    if type(param) in (tuple, list):
        return param[0]
    else:
        return param


def get_freq_value(param):
    """
    Extracts the parameter referring to the
    frequency axis.

    Parameters
    ----------
    param : tuple or scalar
        Module parameter

    Returns
    -------
    scalar
        Temporal parameter
    """
    if type(param) in (tuple, list):
        return param[1]
    else:
        return param


def get_causal_conv_padding(kernel_size: int, dilation: int) -> int:
    """
    Calculates the causal convolutional padding.

    Returns
    -------
    int
       Total causal padding
    """
    causal_pad = (kernel_size - 1) * dilation
    return causal_pad


def get_default_dilation(
    kernel_size: OneOrPair[int],
    depth: int,
    disable_dilation_f: bool = False,
) -> List[Tuple[int, int]]:
    """
    The default dilation is an increasing power of the kernel
    # e.g: d_t_conv2 = k_t**2

    Parameters
    ----------
    kernel_size : OneOrPair[int]
        Size of the kernel
    depth : int
        Number of layers with dilation
    disable_dilation_f : bool, optional
        If True the dilation over frequency will be always 1,
        by default False

    Returns
    -------
    List[Tuple[int, int]]
        default dilation
    """
    k_t, k_f = [f(kernel_size) for f in (get_time_value, get_freq_value)]
    dilation = [(k_t**x, k_f**x) for x in range(depth)]
    if disable_dilation_f:
        dilation = [(d[0], 1) for d in dilation]
    return dilation


def get_causal_longformer_mask(sequence_len: int, receptive_field: int) -> Tensor:
    """
    Obtain a causal mask inspired by the Longformer to use
    within self attention layers.

    https://arxiv.org/abs/2004.05150

    Parameters
    ----------
    sequence_len : int
        Length of the sequence
    receptive_field : int
        Number of previous frames to pay attention to

    Returns
    -------
    Tensor
        Attention mask of shape (sequence_len, sequence_len)
    """
    _diag = lambda i: torch.diag(torch.ones(sequence_len - i), -i)
    mask = _diag(0)
    for i in range(1, receptive_field):
        mask += _diag(i)
    return mask


# utilitiy layers  = = = = = = = = = = = = = = = = = = =
class LambdaLayer(Module):
    def __init__(self, f: Callable):
        """
        Inspired to TF lambda layer

        Parameters
        ----------
        f : Callable
            Callable function
        """
        super(LambdaLayer, self).__init__()
        self.f = f

    def forward(self, *x):
        return self.f(*x)


class LookAhead(Module):
    def __init__(
        self,
        lookahead: int,
        maintain_shape: bool = False,
    ):
        """
        Temporal lookahead layer.
        Input shape: (B, C, T, F)
        Output shape: (B, C, T', F)

        Parameters
        ----------
        lookahead : int
            Amount of lookahead
        maintain_shape : bool, optional
            If set to True, right zero padding is add to compensate
            for the lookahead, by default False
        """
        super().__init__()

        # attributes
        self.lookahead = lookahead
        self.maintain_shape = maintain_shape

        # inner modules
        if self.maintain_shape:
            self.lookahead_pad = nn.ConstantPad2d(
                (0, 0, -self.lookahead, self.lookahead), 0
            )
        else:
            self.lookahead_pad = nn.ConstantPad2d((0, 0, -self.lookahead, 0), 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lookahead_pad(x)
        return x


class Reparameterize(Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample
        from N(mu, var) without breaking the gradient path.

        Parameters
        ----------
        mu : Tensor
            Mean of the latent normal
        logvar : Tensor
            Standard deviation of the latent normal in log scale

        Returns
        -------
        Tensor
            Sample from the normal distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class ScaleChannels2d(Module):
    def __init__(
        self,
        in_channels: int,
    ):
        """
        Learns a per-channel gain.

        Parameters
        ----------
        in_channels : int
            Number of channels
        """
        super().__init__()

        # attributes
        self.in_channels = in_channels

        # inner modules
        self.scale = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            groups=self.in_channels,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a gain to the channels of the input.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, T, F)

        Returns
        -------
        Tensor
            Scaled input of shape (B, C, T, F)
        """
        x = self.scale(x)
        return x


class UnfoldSpectrogram(Module):
    def __init__(self, block_size: int, stride: int):
        """
        Custom unfold module, it performs a windowing over time of a time-freq signal.

        Parameters
        ----------
        block_size : int
            Size of the sliding window over time
        stride : int
            Stride of the sliding window over time
        """
        super().__init__()

        # attributes
        self.block_size = block_size
        self.stride = stride

        # inner modules ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        # (B, C, T, F) -> (B*C, 1, T, F)
        self._reshape_0 = lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])[
            :, None, :, :
        ]

        # (B*C, 1, T, F) -> (B*C, F*block_size, num_blocks)
        self._unfold = lambda x: F.unfold(
            x, (self.block_size, x.shape[3]), stride=(self.stride, 1)
        )

        # (B*C, block_size*F, num_blocks) -> (B*C, 1, block_size, F, num_blocks)
        self._reshape_1 = lambda x: x.reshape(
            x.shape[0], 1, self.block_size, -1, x.shape[2]
        )

        # (B*C, 1, block_size, F, num_blocks) -> (B*C, 1, num_blocks, block_size, F)
        self._reshape_2 = lambda x: x.permute(0, 1, 4, 2, 3)

        # (B*C, 1, num_blocks, block_size, F) -> (B, C, num_blocks, block_size, F)
        self._reshape_3 = lambda x, ch: x.reshape(
            -1, ch, x.shape[2], self.block_size, x.shape[4]
        )

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, T, F)

        Returns
        -------
        Tensor
            Unfolded input of shape (B, C, num_blocks, block_size, F)
        """
        assert self._input_shape_is_valid(x), r"(x.shape[2] - block_size) % stride != 0"
        ch = x.shape[1]
        x = self._reshape_0(x)
        x = self._unfold(x)
        x = self._reshape_1(x)
        x = self._reshape_2(x)
        x = self._reshape_3(x, ch)
        return x

    def _input_shape_is_valid(self, x: Tensor) -> bool:
        """
        Verifies that the input shape is valid for unfolding

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        bool
            True if the input shape is valid
        """
        t = x.shape[2]
        flag = (t - self.block_size) % self.stride == 0
        return flag


class FoldSpectrogram(Module):
    def __init__(self, block_size: int, stride: int, channels: int):
        """
        Custom fold module, reverses the result of UnfoldSpectrogram.

        Parameters
        ----------
        block_size : int
            Size of the sliding window over time
        stride : int
            Stride of the sliding window over time
        channels : int
            Number of channels before UnfoldSpectrogram
        """
        super().__init__()

        # attributes
        self.block_size = block_size
        self.stride = stride
        self.channels = channels

        # inner modules

        # (B, C*num_blocks, block_size, F)
        # used to get the output_size for the folding
        self._get_out_size = lambda x: (
            x.shape[1] * self.stride // self.channels + self.block_size - self.stride,
            x.shape[3],
        )

        # inner modules ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        # (B, C, num_blocks, block_size, F) -> (B, C*num_blocks, block_size, F)
        self._reshape_0 = lambda x: x.reshape(x.shape[0], -1, *x.shape[3:])

        # (B, C*num_blocks, block_size, F) -> (B*C, 1, num_blocks, block_size, F)
        self._reshape_1 = lambda x: x.reshape(
            -1, 1, x.shape[1] // self.channels, self.block_size, x.shape[3]
        )

        # (B*C, 1, num_blocks, block_size, F) -> (B*C, 1, block_size, F, num_blocks)
        self._reshape_2 = lambda x: x.permute(0, 1, 3, 4, 2)

        # (B*C, 1, block_size, F, num_blocks) -> (B*C, block_size*F, num_blocks)
        self._reshape_3 = lambda x: x.reshape(x.shape[0], -1, x.shape[4])

        # (B*C, block_size*F, num_blocks) -> (B*C, 1, T, F)
        self._fold = lambda x, o: F.fold(
            x, o, (self.block_size, o[1]), stride=(self.stride, 1)
        )

        # (B*C, 1, T, F) -> (B, C, T, F)
        self._reshape_4 = lambda x: x.reshape(-1, self.channels, *x.shape[2:])

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, num_blocks, block_size, F)

        Returns
        -------
        Tensor
            Folded input of shape (B, C, T, F)
        """
        x = self._reshape_0(x)
        out_sizes = self._get_out_size(x)
        x = self._reshape_1(x)
        x = self._reshape_2(x)
        x = self._reshape_3(x)
        x = self._fold(x, out_sizes)
        x = self._reshape_4(x)
        divisor = self._get_normalization_tensor(x, out_sizes)
        x /= divisor
        return x

    def _get_normalization_tensor(
        self, x: Tensor, out_sizes: Tuple[int, int]
    ) -> Tensor:
        """
        Obtain the normalization tensor for perfect fold/unfold inversion.
        https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Unfold

        Parameters
        ----------
        x : Tensor
            Input tensor
        out_sizes : Tuple[int, int]
            Parameter output_size needed for folding


        Returns
        -------
        Tensor
            Normalization tensor
        """
        o = out_sizes
        y = torch.ones_like(x)
        y = F.unfold(y, (self.block_size, o[1]), stride=(self.stride, 1))
        y = F.fold(y, o, (self.block_size, o[1]), stride=(self.stride, 1))
        return y


class ResidualWrap(Module):
    def __init__(self, *modules: List[Module]):
        """
        Given a series of layers F(x) it reparametrize
        them as (x + F(x)).

        F(x) and x shoud have compatible shapes.

        Parameters
        ----------
        modules : List[Module]
            Series of modules
        """
        super().__init__()
        self.layers = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        # keeping the channels of the output
        c_out = y.shape[1]
        y = x[:, :c_out] + y[:, :c_out]

        return y


# dense variants = = = = = = = = = = = = = = = = = = = =
class GroupedLinear(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        groups: int = 1,
    ):
        """
        Dense layer with grouping.

        Parameters
        ----------
        input_dim : int
            Input feature size
        output_dim : int
            Output feature size
        groups : int, optional
            number of groups, by default 1
        """
        super().__init__()

        # attributes
        self.input_dim = input_dim
        self.hidden_size = output_dim
        self.ws = input_dim // groups
        self.groups = groups

        # error handling
        assert (
            input_dim % groups == 0
        ), f"Input size {input_dim} not divisible by {groups}"
        assert (
            output_dim % groups == 0
        ), f"Hidden size {output_dim} not divisible by {groups}"

        # weights
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.zeros(groups, input_dim // groups, output_dim // groups),
                requires_grad=True,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the dense weights.
        """
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies self.groups different dense layers
        distributed per sections.

        Parameters
        ----------
        x : Tensor
            Input of shape (..., self.input_dim)

        Returns
        -------
        Tensor
            Output of shape (..., self.hidden_size)
        """
        new_shape = (*x.shape[:-1], self.groups, self.ws)
        x = x.reshape(new_shape)
        x = torch.einsum("...gi,gih->...gh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(-2, -1)
        return x

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_dim: {self.input_dim}, hidden_size: {self.hidden_size}, groups: {self.groups})"


# conv2d variants = = = = = = = = = = = = = = = = = = =
class CausalConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: OneOrPair[int],
        stride_f: int = 1,
        padding_f: Optional[int] = None,
        dilation: OneOrPair[int] = 1,
        bias: bool = True,
        separable: bool = False,
        dtype=None,
    ):
        """
        Convolution with causal kernels over time

        Parameters
        ----------
        Same parameters as Conv2d plus

        stride_f : int
            Kernel stride over frequency
        padding_f : Optional[int]
            Symmetric padding over frequency, by default ensures f_out = f_in // stride_f
            if stride_f divides f_in
        separable : bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        """
        super().__init__()

        # attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride_f = stride_f
        self.padding_f = padding_f
        self.dilation = dilation
        self.bias = bias
        self.separable = separable
        self.dtype = dtype

        # inner modules
        self.causal_pad_amount = self._get_causal_pad_amount()
        self.groups = np.gcd(self.in_channels, self.out_channels)

        layers = []

        # causal padding
        causal_pad = nn.ConstantPad2d((0, 0, self.causal_pad_amount, 0), 0)
        layers.append(causal_pad)

        # freq padding
        freq_pad = self._get_default_freq_padding()
        layers.append(freq_pad)

        # convolution
        conv_pad = 0 if self.padding_f is None else (0, self.padding_f)
        if not self.separable:
            full_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=(1, stride_f),
                dilation=self.dilation,
                bias=self.bias,
                dtype=self.dtype,
                padding=conv_pad,
            )

            layers.append(full_conv)
        else:
            # separable convolution
            # depthwise + pointwise
            depthwise = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=(1, stride_f),
                dilation=self.dilation,
                bias=self.bias,
                groups=self.groups,
                dtype=self.dtype,
                padding=conv_pad,
            )

            pointwise = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                bias=False,
                dtype=self.dtype,
            )

            layers += [depthwise, pointwise]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x

    def _get_causal_pad_amount(self) -> int:
        """
        Calculates the causal padding.
        """
        kernel_size_t, dilation_t = map(
            get_time_value, (self.kernel_size, self.dilation)
        )
        causal_pad = get_causal_conv_padding(kernel_size_t, dilation_t)
        return causal_pad

    def _get_default_freq_padding(self) -> Module:
        """
        Gets the default frequency padding.

        Returns
        -------
        Module
            Frequency padding module
        """
        kernel_size_f, dilation_f = map(
            get_freq_value, (self.kernel_size, self.dilation)
        )
        pad_f = dilation_f * (kernel_size_f - 1) + 1 - self.stride_f
        half_pad_f = pad_f // 2
        if self.padding_f is not None:
            pad = nn.Identity()  # manual padding
        elif pad_f % 2 == 0:
            pad = nn.ConstantPad2d((half_pad_f, half_pad_f, 0, 0), 0)
        else:
            pad = nn.ConstantPad2d((half_pad_f + 1, half_pad_f, 0, 0), 0)
        return pad


class CausalSubConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: OneOrPair[int],
        stride: int = 1,
        dilation: OneOrPair[int] = 1,
        bias: bool = True,
        separable: bool = False,
        dtype=None,
    ):
        """
        Also known as sub pixel convolution, described in
        Dense CNN With Self-Attention for Time-Domain Speech Enhancement
        paper (https://ieeexplore.ieee.org/document/9372863).

        Many convolutions are interleaved to upsample a signal over frequency or time.

        Parameters
        ----------
        Same parameters as Conv2d plus

        stride : int
            Defines how many interleaved convolutions are present
        separable : bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        """
        super().__init__()

        # attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.separable = separable
        self.dtype = dtype

        # inner modules
        _conv = lambda: CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride_f=1,
            padding_f=None,
            dilation=dilation,
            bias=bias,
            separable=separable,
            dtype=dtype,
        )
        self.layers = nn.Sequential(*[_conv() for _ in range(self.stride)])
        self.transpose = lambda x: x.transpose(2, 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, T, F)

        Returns
        -------
        Tensor
            Output of shape (B, C, T, F * stride)
        """
        xs = [conv(x) for conv in self.layers]
        x = interleave(*xs)
        return x


# conv2d compositions = = = = = = = = = = = = = = = = =
class CausalConv2dNormAct(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: OneOrPair[int],
        stride_f: int = 1,
        dilation: OneOrPair[int] = 1,
        separable: bool = False,
        batchnorm_eps: float = 1e-05,
        batchnorm_momentum: float = 0.1,
        batchnorm_affine: bool = True,
        batchnorm_track_running_stats: bool = True,
        activation: Module = nn.ReLU(),
        residual_merge: Optional[Callable] = None,
        merge_after_conv: bool = True,
        disable_batchnorm: bool = False,
        dtype=None,
    ):
        """
        CausalConv2d + BatchNorm2d + Activation.

        This layer ensures f_in = f_out // stride_f only if stride_f divides f_in.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : OneOrPair[int]
            Convolution kernel size
        stride_f : int, optional
            Frequency stride, by default 1
        dilation : OneOrPair[int], optional
            Convolution dilation, by default 1
        separable : bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        batchnorm_eps : float, optional
            Eps parameter of the BatchNorm2d , by default 1e-05
        batchnorm_momentum : float, optional
            Momentum parameter of the BatchNorm2d, by default 0.1
        batchnorm_affine : bool, optional
            Affine parameter of the BatchNorm2d, by default True
        batchnorm_track_running_stats : bool, optional
            Track running stats parameter of the BatchNorm2d, by default True
        activation : Module, optional
            Activation module, by default nn.Relu()
        residual_merge : Optional[Callable], optional
            If different da None, it indicates the merge operation using a skip connection
            from the output of the transposed conv to the output of the activation, by default None
        merge_after_conv : bool, optional
            If True one the input used in the residual_merge layer is taken after the CausalConv2d,
            instead that from the input directly, by default True
        disable_batchnorm : bool, optional
            Disable the BatchNorm2d layer, by default False
        dtype, optional
            Module dtype, by default None
        """

        super().__init__()

        # attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride_f = stride_f
        self.dilation = dilation
        self.separable = separable
        self.batchnorm_eps = batchnorm_eps
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_affine = batchnorm_affine
        self.batchnorm_track_running_stats = batchnorm_track_running_stats
        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge
        self.merge_after_conv = merge_after_conv
        self.disable_batchnorm = disable_batchnorm
        self.dtype = dtype

        # inner modules
        self.conv = CausalConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride_f=stride_f,
            dilation=self.dilation,
            bias=False,
            separable=self.separable,
            dtype=self.dtype,
        )

        if self.disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm2d(
                num_features=self.out_channels,
                eps=self.batchnorm_eps,
                momentum=self.batchnorm_momentum,
                affine=self.batchnorm_affine,
                track_running_stats=self.batchnorm_track_running_stats,
                dtype=self.dtype,
            )

    def forward(self, x: Tensor) -> Tensor:
        z = x
        x = self.conv(x)
        y = self.batchnorm(x)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x if self.merge_after_conv else z, y)
        return y


class DenseConvBlock(Module):
    def __init__(
        self,
        channels: int,
        kernel_size: OneOrPair[int],
        feature_size: int,
        dilation: Optional[OneOrPair[int]] = None,
        disable_dilation_f: bool = False,
        depth: int = 3,
        final_stride: int = 1,
        batchnorm_eps: float = 1e-05,
        batchnorm_affine: bool = True,
        disable_layernorm: bool = False,
        activation: Module = nn.LeakyReLU(),
        dtype=None,
    ):
        """
        Dense block from Dense CNN With Self-Attention for Time-Domain Speech Enhancement
        paper (https://ieeexplore.ieee.org/document/9372863).

        Parameters
        ----------
        channels : int
            Number of input and output channels
        kernel_size : OneOrPair[int]
            Size of the convolutional kernels
        feature_size : int
            Size of the last dimension
        dilation : Optional[Tuple[int,int]]
            By default the dilation is equal to the kernel to the power of
            the post_conv layer index
        disable_dilation_f : bool
            If True dilation_f==1 for each conv dilation setting,
            by default False
        tconv_kernel_f : Optional[int], optional
            Frequncy kernel size of the transposed convolution,
            by default twice tconv_stride_f
        final_stride : int, optional
            Stride of the last convolution, by default 1
        batchnorm_eps : float, optional
            Eps parameter of the BatchNorm2d , by default 1e-05
        batchnorm_affine : bool, optional
            Affine parameter of the BatchNorm2d, by default True
        disable_layernorm : bool, optional
            Disables the batch normalization, by default False
        activation : Module, optional
            Activation to use, by default nn.LeakyReLU()
        dtype : optional
            Module dtype, by default None
        """
        super().__init__()

        # attributes
        self.channels = channels
        self.kernel_size = kernel_size
        self.feature_size = feature_size
        self.dilation = dilation
        self.disable_dilation_f = disable_dilation_f
        self.depth = depth
        self.final_stride = final_stride
        self.batchnorm_eps = batchnorm_eps
        self.batchnorm_affine = batchnorm_affine
        self.disable_layernorm = disable_layernorm
        self.activation = activation or nn.Identity()
        self.dtype = dtype

        # default dilation ~ ~ ~ ~ ~ ~
        self.dilation = (
            get_default_dilation(self.kernel_size, self.depth, self.disable_dilation_f)
            if self.dilation is None
            else [self.dilation] * self.depth
        )
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        # inner modules
        _sample_norm = lambda i: (
            nn.Identity()
            if (self.disable_layernorm or i == (self.depth - 1))
            else nn.LayerNorm(
                normalized_shape=self.feature_size,
                eps=self.batchnorm_eps,
                elementwise_affine=self.batchnorm_affine,
                dtype=self.dtype,
            )
        )

        _block = lambda i, k, d: nn.Sequential(
            CausalConv2dNormAct(
                in_channels=(i + 1) * self.channels,
                out_channels=self.channels,
                kernel_size=k,
                stride_f=self.final_stride if i == (self.depth - 1) else 1,
                dilation=d,
                separable=False,
                activation=None,
                disable_batchnorm=True,
                dtype=self.dtype,
            ),
            _sample_norm(i),
            self.activation,
        )

        kernels = [self.kernel_size] * self.depth
        layers = [
            _block(i, k, d) for i, (k, d) in enumerate(zip(kernels, self.dilation))
        ]

        self.layers = nn.Sequential(*layers)

        # concat over channels
        self._concat = lambda x, y: torch.cat((x, y), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Signal of shape (B, C, T, F)

        Returns
        -------
        Tensor
            Output of shape (B, C, T, F//final_stride)
        """
        for i, block in enumerate(self.layers):
            y = block(x)
            x = self._concat(x, y) if i != (self.depth - 1) else y

        return x


# recurrent layers variants = = = = = = = = = = = = = =
class GruNormAct(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        gru_bias: bool = False,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: Module = nn.ReLU(),
        residual_merge: Optional[Callable] = None,
        disable_batchnorm: bool = False,
        dtype=None,
    ):
        """
        GRU + BatchNorm1d + Activation.

        Parameters
        ----------
        Combination of the modules parameters

        activation: Module, optional
            Activation module, by default nn.Relu()
        residual_merge: Optional[Callable], optional
            If different da None, it indicates the merge operation after
            the activation, by default None
        disable_batchnorm: bool, optional
            Disable the BatchNorm2d layer, by default False
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_bias = gru_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge
        self.disable_batchnorm = disable_batchnorm
        self.dtype = dtype

        # inner modules
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.gru_bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            dtype=self.dtype,
        )

        if self.disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm1d(
                num_features=self.hidden_size,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
                dtype=self.dtype,
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y, h = self.gru(x)
        y = y.transpose(1, 2)
        y = self.batchnorm(y)
        y = y.transpose(1, 2)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y, h


# attention variants  = = = = = = = = = = = = = = = = =


# This is untested
class SlidingCausalMultiheadAttention(Module):
    def __init__(
        self,
        channels: int,
        sequence_len: int,
        embed_dim: int,
        stride: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        receptive_field: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Causal multi head attention module applied on an
        unfolded version of the input.

        Parameters
        ----------
        channels : int
            Number of input channels
        embed_dim : int
            Dimension of the embeddings (F)
        sequence_len : int
            Lenght of the sequence blocks resulting from the unfolding
        stride : int
            Hop size between the blocks
        num_heads : int
            Number of heads for the attention
        dropout : float, optional
            Dropout amount, by default 0.0
        bias : bool, optional
            Enables/disables the biases in the projections, by default True
        receptive_field : Optional[int], optional
            Used for the default causal Longformer mask, indicates the number of frames
            employed in the computation of the attention, by default half of the sequence_len
        attn_mask : Optional[Tensor], optional
            Mask provided to the attention layer, by default a causal version of the Longformer
            mask is provided https://arxiv.org/abs/2004.05150
        """
        super().__init__()

        # attributes
        self.channels = channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sequence_len = sequence_len
        self.stride = stride
        self.dropout = dropout
        self.bias = bias
        self.receptive_field = receptive_field or (self.sequence_len // 2)
        self.attn_mask = attn_mask or get_causal_longformer_mask(
            self.sequence_len, self.receptive_field
        )

        # inner modules
        self.reshape_0 = lambda x: x.flatten(0, 2)
        self.reshape_1 = lambda x, b, c: x.reshape(b, c, -1, *x.shape[-2:])

        self.unfold = UnfoldSpectrogram(self.sequence_len, self.stride)
        self.fold = FoldSpectrogram(self.sequence_len, self.stride, self.channels)

        self.mh_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=self.bias,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, T, F)

        Returns
        -------
        Tensor
            Output of shape (B, C, T, F)
        """
        b, c = x.shape[:2]
        x = self.unfold(x)
        x = self.reshape_0(x)
        x = self.mh_attention(x, x, x, attn_mask=self.attn_mask)[0]
        x = self.reshape_1(x, b, c)
        x = self.fold(x)
        return x


# = = = = = = = = = = = = = = = = = = = = = = = = = = =
