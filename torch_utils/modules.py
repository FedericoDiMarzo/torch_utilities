from typing import Callable, List, Optional, Tuple
from torch.nn.utils import weight_norm
from pathimport import set_module_root
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import torch

set_module_root(".")

__all__ = [
    "LambdaLayer",
    "Lookahead",
    "Reparameterize",
    "ScaleChannels2d",
    "GroupedLinear",
    "CausalConv2d",
    "GruNormAct",
    "CausalConv2dNormAct",
    "CausalConvNeuralUpsampler",
    "DownMerge",
    "UpMerge",
]


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
    if isinstance(param, tuple):
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
    if isinstance(param, tuple):
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


class LambdaLayer(nn.Module):
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

    def forward(self, x):
        return self.f(x)


class Lookahead(nn.Module):
    def __init__(
        self,
        lookahead: int,
        maintain_shape: bool = False,
    ) -> None:
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

        self.lookahead = lookahead
        self.maintain_shape = maintain_shape

        if self.maintain_shape:
            self.lookahead_pad = nn.ConstantPad2d((0, 0, -self.lookahead, self.lookahead), 0)
        else:
            self.lookahead_pad = nn.ConstantPad2d((0, 0, -self.lookahead, 0), 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lookahead_pad(x)
        return x


class Reparameterize(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample
        from N(mu, var) from N(0,1).

        Parameters
        ----------
        mu : Tensor
            Mean of the latent normal
        logvar : Tensor
            Standard deviation of the latent normal in log

        Returns
        -------
        Tensor
            Sample from the normal distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class ScaleChannels2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        """
        Learns a per-channel gain.

        Parameters
        ----------
        in_channels : int
            Number of channels
        """
        super().__init__()

        # inner modules
        self.scale = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=in_channels,
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


class GroupedLinear(nn.Module):
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
        self.input_dim = input_dim
        self.hidden_size = output_dim
        self.ws = input_dim // groups
        self.groups = groups

        # error handling
        assert input_dim % groups == 0, f"Input size {input_dim} not divisible by {groups}"
        assert output_dim % groups == 0, f"Hidden size {output_dim} not divisible by {groups}"

        # weights
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.zeros(groups, input_dim // groups, output_dim // groups),
                requires_grad=True,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
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
            Input og shape (..., self.input_dim)

        Returns
        -------
        Tensor
            _description_
        """
        # x: [..., I]
        b, t = x.shape[:2]
        # new_shape = list(x.shape)[:-1] + [self.groups, self.ws]
        new_shape = (b, t, self.groups, self.ws)
        x = x.view(new_shape)
        # The better way, but not supported by torchscript
        # x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
        x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(2, 3)  # [B, T, H]
        return x

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_dim: {self.input_dim}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride_f: int = 1,
        padding_f: int = 0,
        dilation: Tuple[int, int] = 1,
        bias: bool = True,
        separable: bool = False,
        enable_weight_norm: bool = False,
        dtype=None,
    ) -> None:
        """
        Convolution with causal kernels over time

        Parameters
        ----------
        Same parameters as Conv2d plus

        separable: bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        enable_weight_norm : bool, optional
            Enables weight normalization, by default False
        """
        super().__init__()
        self.causal_pad_amount = self._get_causal_pad_amount(kernel_size, dilation)
        self.enable_weight_norm = enable_weight_norm
        self.separable = separable

        # optional weight normalization
        self._normalize = weight_norm if self.enable_weight_norm else (lambda x: x)

        # inner modules
        self.causal_pad = nn.ConstantPad2d((0, 0, self.causal_pad_amount, 0), 0)
        groups = np.gcd(in_channels, out_channels)

        if not self.separable:
            self.conv = self._normalize(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, stride_f),
                    padding=(0, padding_f),
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                )
            )
        else:
            # separable convolution
            # depthwise + pointwise
            depthwise = self._normalize(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, stride_f),
                    padding=(0, padding_f),
                    dilation=dilation,
                    bias=bias,
                    groups=groups,
                    dtype=dtype,
                )
            )
            pointwise = self._normalize(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    dtype=dtype,
                )
            )
            self.conv = nn.Sequential(depthwise, pointwise)

    def forward(self, x: Tensor) -> Tensor:
        x = self.causal_pad(x)
        x = self.conv(x)
        return x

    def _get_causal_pad_amount(self, kernel_size: int, dilation: int) -> int:
        """
        Calculates the causal padding.
        """
        kernel_size, dilation = map(get_time_value, (kernel_size, dilation))
        causal_pad = get_causal_conv_padding(kernel_size, dilation)
        return causal_pad


class CausalConv2dNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride_f: int = 1,
        dilation: Tuple[int, int] = 1,
        separable: bool = False,
        batchnorm_eps: float = 1e-05,
        batchnorm_momentum: float = 0.1,
        batchnorm_affine: bool = True,
        batchnorm_track_running_stats: bool = True,
        activation: nn.Module = nn.ReLU(),
        residual_merge: Optional[Callable] = None,
        disable_batchnorm: bool = False,
        enable_weight_norm: bool = False,
        dtype=None,
    ) -> None:
        """
        CausalConv2d + BatchNorm2d + Activation.

        This layer ensures f_in = f_out // stride_f only if
        f_in is even, stride_f divides f_in and
        [dilation_f * (kernel_f - 1) + 1] is odd.

        Otherwise f_in = f_out // stride_f + 1.


        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : Tuple[int, int]
            Convolution kernel size
        stride_f : int, optional
            Frequency stride, by default 1
        dilation : Tuple[int, int], optional
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
        activation : nn.Module, optional
            Activation module, by default nn.Relu()
        residual_merge : Optional[Callable], optional
            If different da None, it indicates the merge operation after
            the activation, by default None
        disable_batchnorm : bool, optional
            Disable the BatchNorm2d layer, by default False
        enable_weight_norm : bool, optional
            Uses the weight normalization instead of the batch normalization,
            by default False
        dtype : _type_, optional
            Module dtype, by default None
        """

        super().__init__()
        self.separable = separable
        self.disable_batchnorm = disable_batchnorm
        self.enable_weight_norm = enable_weight_norm

        # weight normalization disables batch normalization
        if self.enable_weight_norm:
            self.disable_batchnorm = True

        # inner modules
        kernel_f = get_freq_value(kernel_size)
        dilation_f = get_freq_value(dilation)
        half_padding_f = self._get_freq_padding(kernel_f, dilation_f)

        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding_f=half_padding_f,
            stride_f=stride_f,
            dilation=dilation,
            bias=False,
            separable=separable,
            enable_weight_norm=self.enable_weight_norm,
            dtype=dtype,
        )

        if self.disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm2d(
                num_features=out_channels,
                eps=batchnorm_eps,
                momentum=batchnorm_momentum,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_running_stats,
                dtype=dtype,
            )

        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        y = self.batchnorm(y)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y

    def _get_freq_padding(self, kernel_f: int, dilation_f: int) -> int:
        """
        Gets the padding needed to keep the frequency output dimension
        equal to f_out = int(f_in / stride_f + 1).

        The previous formula it's valid only when
        [dilation_f * (kernel_f - 1) + 1 / 2] is an integer.
        (more information here https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

        Parameters
        ----------
        kernel_f : int
            Kernel size over frequency dimension
        dilation_f : int
            Dilation over frequency dimension
        stride_f : int
            Stride value over frequency dimension

        Returns
        -------
        int
            Padding needed to approximate f_out = int(f_in / stride_f + 1)
        """

        padding = get_causal_conv_padding(kernel_f, dilation_f) + 1
        padding = padding // 2  # the approximation occours here
        return padding


class CausalConvNeuralUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        post_conv_kernel_size: Tuple[int, int],
        post_conv_count: int = 1,  # TODO: implement it
        post_conv_dilation: Tuple[int, int] = 1,
        tconv_kernel_f_size: Optional[int] = None,
        tconv_stride_f: int = 2,
        tconv_padding_f: int = 0,
        separable: bool = False,
        batchnorm_eps: float = 1e-05,
        batchnorm_momentum: float = 0.1,
        batchnorm_affine: bool = True,
        batchnorm_track_running_stats: bool = True,
        disable_batchnorm: bool = False,
        enable_weight_norm: bool = False,
        activation: nn.Module = nn.LeakyReLU(),
        residual_merge: Optional[Callable] = None,
        dtype=None,
    ) -> None:
        """
        Frequency upsampling module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        post_conv_kernel_size : Tuple[int, int]
            Kernel size of the post convolutions. Can be also an int, or a list
            of ints/Tuple[int, int] of length post_conv_count
        post_conv_count : int, optional
            Number of post convolutions, by default 1
        tconv_kernel_f_size : Optional[int], optional
            Frequncy kernel size of the transposed convolution,
            by default twice tconv_stride_f
        tconv_stride_f : int, optional
            Stride of the transposed convolution, by default 2
        tconv_padding_f : int, optional
            Frequency padding of the transposed convolution , by default 0
        separable : bool, optional
            Enable separable convolutions, by default False
        batchnorm_eps : float, optional
            Eps parameter of the BatchNorm2d , by default 1e-05
        batchnorm_momentum : float, optional
            Momentum parameter of the BatchNorm2d, by default 0.1
        batchnorm_affine : bool, optional
            Affine parameter of the BatchNorm2d, by default True
        batchnorm_track_running_stats : bool, optional
            Track running stats parameter of the BatchNorm2d, by default True
        disable_batchnorm : bool, optional
            Disables the batch normalization, by default False
        enable_weight_norm : bool, optional
            Uses the weight normalization instead of the batch normalization,
            by default False
        activation : nn.Module, optional
            Activation to use, by default nn.LeakyReLU()
        residual_merge : Optional[Callable], optional
            Residual skip connection, by default None
        dtype : optional
            Module dtype, by default None
        """
        super().__init__()
        self.separable = separable
        self.disable_batchnorm = disable_batchnorm
        self.enable_weight_norm = enable_weight_norm

        # optional weight normalization
        self._normalize = weight_norm if self.enable_weight_norm else (lambda x: x)

        # weight normalization disables batch normalization
        if self.enable_weight_norm:
            self.disable_batchnorm = True

        # tconv_kernel_f_size is set to twice tconv_stride_f
        # to reduce the reconstruction artifacts
        if tconv_kernel_f_size is None:
            tconv_kernel_f_size = 2 * tconv_stride_f

        # inner modules
        self.tconv = self._normalize(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, tconv_kernel_f_size),
                stride=(1, tconv_stride_f),
                padding=(0, tconv_padding_f),
                output_padding=(0, tconv_stride_f - 1),
                bias=False,
                dtype=dtype,
            )
        )

        pad_f = tconv_kernel_f_size // 2
        if tconv_kernel_f_size % 2 == 0:
            # even tconv kernel
            self.padding_f = nn.ConstantPad2d((-pad_f, -pad_f + 1, 0, 0), 0)
        else:
            self.padding_f = nn.ConstantPad2d((-pad_f, -pad_f, 0, 0), 0)

        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=post_conv_kernel_size,
            padding_f=post_conv_kernel_size[1] // 2,
            dilation=post_conv_dilation,
            bias=False,
            separable=separable,
            enable_weight_norm=self.enable_weight_norm,
            dtype=dtype,
        )

        if disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm2d(
                num_features=out_channels,
                eps=batchnorm_eps,
                momentum=batchnorm_momentum,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_running_stats,
                dtype=dtype,
            )

        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tensor:
        y = self.tconv(x)
        y = self.padding_f(y)
        y = self.conv(y)
        y = self.batchnorm(y)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y


class GruNormAct(nn.Module):
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
        activation: nn.Module = nn.ReLU(),
        residual_merge: Optional[Callable] = None,
        disable_batchnorm: bool = False,
        dtype=None,
    ) -> None:
        """
        GRU + BatchNorm1d + Activation.

        Parameters
        ----------
        Combination of the modules parameters

        activation: nn.Module, optional
            Activation module, by default nn.Relu()
        residual_merge: Optional[Callable], optional
            If different da None, it indicates the merge operation after
            the activation, by default None
        disable_batchnorm: bool, optional
            Disable the BatchNorm2d layer, by default False
        """
        super().__init__()
        self.activation = activation
        self.residual_merge = residual_merge
        self.disable_batchnorm = disable_batchnorm

        # inner modules
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=gru_bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            dtype=dtype,
        )

        if self.disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm1d(
                num_features=hidden_size,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                dtype=dtype,
            )

        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y, h = self.gru(x)
        y = y.transpose(1, 2)
        y = self.batchnorm(y)
        y = y.transpose(1, 2)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y, h


class DownMerge(nn.Module):
    def __init__(self, channels: int, scaling: int) -> None:
        """
        Merge node that can be associated
        to CausalConv2dNormAct o similar encoding nodes.

        Parameters
        ----------
        channels : int
            Output channels of a encoding module
        scaling : int
            Stride of a encoding module
        """
        super().__init__()
        self.channels = channels
        self.scaling = scaling

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input before the encoding module
        y : Tensor
            Input after the encoding module

        Returns
        -------
        Tensor
            Merged thensor with the same shape as y
        """
        # reducing the frequencies
        x = x[..., :: self.scaling]
        c_in = x.shape[1]
        c_out = self.channels

        # increasing the channels
        err_msg = f"c_in=={c_in} should divide c_out=={c_out},"
        ratio = c_out // c_in
        assert ratio > 0, err_msg
        if ratio > 0:
            x = torch.cat([x] * ratio, dim=1)

        # merging
        x = x + y

        return x


class UpMerge(nn.Module):
    def __init__(self, channels: int, scaling: int) -> None:
        """
        Merge node that can be associated to
        CausalConvNeuralUpsampler o similar decoding modules.

        Parameters
        ----------
        channels : int
            Output channels of a decoding module
        scaling : int
            Stride of a decoding module
        """
        super().__init__()
        self.channels = channels
        self.scaling = scaling

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input before the decoding module
        y : Tensor
            Input after the decoding module

        Returns
        -------
        Tensor
            Merged thensor with the same shape as y
        """
        # increasing the frequencies
        x = F.interpolate(x, scale_factor=(1, self.scaling))
        c_in = x.shape[1]
        c_out = self.channels

        # reducing the channels
        err_msg = f"c_out=={c_out} should divide c_in=={c_in}"
        ratio = c_in // c_out
        assert ratio > 0, err_msg
        x = x[:, ::ratio]

        # merging
        x = x + y

        return x
