import torch
import numpy as np
from typing import Callable, Optional, Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from pathimport import set_module_root

set_module_root(".", prefix=True)
from torch_utils.common import get_device

__all__ = [
    "Lookahead",
    "CausalConv2d",
    "CausalConv2dNormAct",
    "CausalConvNeuralUpsampler",
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


class Lookahead(nn.Module):
    def __init__(self, lookahead: int, maintain_shape: bool = False) -> None:
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
        return self.lookahead_pad(x)


class Reparameterize(nn.Module):
    # TODO: tests
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """

        Reparameterization trick to sample
        from N(mu, var) from N(0,1).

        :param mu: (Tensor) Mean of the latent normal [B x D]
        :param logvar: (Tensor) Standard deviation of the latent normal [B x D]
        :return: (Tensor) [B x D]


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


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = 1,
        padding: Tuple[int, int] = 0,
        dilation: Tuple[int, int] = 1,
        bias: bool = True,
        separable: bool = False,
        device=None,
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
        self.causal_pad_amount = self._get_causal_pad_amount(kernel_size, stride, dilation)
        self.separable = separable

        # error handling
        err_msg = "only stride[0] == 1 is supported"
        assert get_time_value(stride) == 1, err_msg
        err_msg = "temporal padding cannot be set explicitely"
        assert get_time_value(padding) == 0, err_msg

        # inner modules
        self.causal_pad = nn.ConstantPad2d((0, 0, self.causal_pad_amount, 0), 0)

        if not self.separable:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        else:
            # separable convolution
            # depthwise + pointwise
            groups = np.gcd(in_channels, out_channels)
            depthwise = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=groups,
                device=device,
                dtype=dtype,
            )
            pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.conv = nn.Sequential(depthwise, pointwise)

    def forward(self, x: Tensor) -> Tensor:
        x = self.causal_pad(x)
        x = self.conv(x)
        return x

    def _get_causal_pad_amount(self, kernel_size, stride, dilation) -> int:
        """
        Calculates the causal padding.
        """
        # TODO: support stride
        kernel_size, stride, dilation = map(get_time_value, (kernel_size, stride, dilation))
        causal_pad = (kernel_size - 1) * dilation
        return causal_pad


class CausalConv2dNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = 1,
        padding: Tuple[int, int] = 0,
        dilation: Tuple[int, int] = 1,
        separable: bool = False,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: nn.Module = nn.ReLU(),
        residual_merge: Optional[Callable] = None,
        disable_batchnorm: bool = False,
        device=None,
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
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            separable=separable,
            device=device,
            dtype=dtype,
        )

        self.batchnorm = nn.BatchNorm2d(
            num_features=out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

        self.activation = activation
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        y = self.batchnorm(y)
        if self.disable_batchnorm:
            y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y


class CausalConvNeuralUpsampler(nn.Module):
    # TODO: tests
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tconv_kernel_size: Tuple[int, int],
        conv_kernel_size: Tuple[int, int],
        tconv_stride: Tuple[int, int] = 2,
        tconv_padding: Tuple[int, int] = 0,
        tconv_output_padding: Tuple[int, int] = 0,
        conv_padding: Tuple[int, int] = 0,
        dilation: Tuple[int, int] = 1,
        separable: bool = False,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: nn.Module = nn.LeakyReLU(),
        residual_merge: Optional[Callable] = None,
        disable_batchnorm: bool = False,
        device=None,
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
            kernel_size=tconv_kernel_size,
            stride=tconv_stride,
            padding=tconv_padding,
            output_padding=tconv_output_padding,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
            dilation=dilation,
            bias=False,
            separable=separable,
            device=device,
            dtype=dtype,
        )
        self.batchnorm = nn.BatchNorm2d(
            num_features=out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

        self.activation = activation
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tensor:
        y = self.tconv(x)
        y = self.conv(y)
        y = self.batchnorm(y)
        if self.disable_batchnorm:
            y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y
