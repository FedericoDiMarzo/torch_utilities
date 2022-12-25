from typing import Callable, List, Optional, Tuple
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
        dtype=None,
    ) -> None:
        """
        Convolution with causal kernels over time

        Parameters
        ----------
        Same parameters as Conv2d plus

        separable: bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        """
        super().__init__()
        self.causal_pad_amount = self._get_causal_pad_amount(kernel_size, dilation)
        self.separable = separable

        # inner modules
        self.causal_pad = nn.ConstantPad2d((0, 0, self.causal_pad_amount, 0), 0)
        groups = np.gcd(in_channels, out_channels)

        if not self.separable:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, stride_f),
                padding=(0, padding_f),
                dilation=dilation,
                bias=bias,
                dtype=dtype,
            )
        else:
            # separable convolution
            # depthwise + pointwise
            depthwise = nn.Conv2d(
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
            pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                dtype=dtype,
            )
            self.conv = nn.Sequential(depthwise, pointwise)

    def forward(self, x: Tensor) -> Tensor:
        x = self.causal_pad(x)
        x = self.conv(x)
        return x

    def _get_causal_pad_amount(self, kernel_size, dilation) -> int:
        """
        Calculates the causal padding.
        """
        kernel_size, dilation = map(get_time_value, (kernel_size, dilation))
        causal_pad = (kernel_size - 1) * dilation
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
        CausalConv2d + BatchNorm2d + Activation.

        Parameters
        ----------
        Combination of the modules parameters

        separable: bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        activation: nn.Module, optional
            Activation module, by default nn.Relu()
        residual_merge: Optional[Callable], optional
            If different da None, it indicates the merge operation after
            the activation, by default None
        disable_batchnorm: bool, optional
            Disable the BatchNorm2d layer, by default False
        """
        super().__init__()
        self.separable = separable
        self.disable_batchnorm = disable_batchnorm

        # inner modules
        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding_f=kernel_size[1] // 2,
            stride_f=stride_f,
            dilation=dilation,
            bias=False,
            separable=separable,
            dtype=dtype,
        )

        if kernel_size[1] % 2 == 0:
            self.freq_trim = nn.ConstantPad2d((0, -1, 0, 0), 0)
        else:
            self.freq_trim = nn.Identity()

        if self.disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm2d(
                num_features=out_channels,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                dtype=dtype,
            )

        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        y = self.freq_trim(y)
        y = self.batchnorm(y)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y


class CausalConvNeuralUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: Tuple[int, int],
        tconv_kernel_f_size: int = 4,
        tconv_stride_f: int = 2,
        tconv_padding_f: int = 0,
        dilation: Tuple[int, int] = 1,
        separable: bool = False,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: nn.Module = nn.LeakyReLU(),
        residual_merge: Optional[Callable] = None,
        disable_batchnorm: bool = False,
        dtype=None,
    ) -> None:
        """
        ConvTranspose2d + CausalConv2d + BatchNorm2d + Activation.

        Parameters
        ----------
        Combination of the modules parameters

        separable: bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        activation: nn.Module, optional
            Activation module, by default nn.Relu()
        residual_merge: Optional[Callable], optional
            If different da None, it indicates the merge operation after
            the activation, by default None
        disable_batchnorm: bool, optional
            Disable the BatchNorm2d layer, by default False
        """
        super().__init__()
        self.separable = separable
        self.disable_batchnorm = disable_batchnorm

        # inner modules
        self.tconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, tconv_kernel_f_size),
            stride=(1, tconv_stride_f),
            padding=(0, tconv_padding_f),
            output_padding=(0, tconv_stride_f - 1),
            bias=False,
            dtype=dtype,
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
            kernel_size=conv_kernel_size,
            padding_f=conv_kernel_size[1] // 2,
            dilation=dilation,
            bias=False,
            separable=separable,
            dtype=dtype,
        )

        if disable_batchnorm:
            self.batchnorm = nn.Identity()
        else:
            self.batchnorm = nn.BatchNorm2d(
                num_features=out_channels,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
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
