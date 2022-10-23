from typing import Callable, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pathimport import set_module_root

set_module_root(".", prefix=True)
from torch_utils.common import get_device

__all__ = ["Lookahead", "CausalConv2d", "CausalConv2dNormAct"]


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


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = 1,
        padding: Tuple[int, int] = 0,
        dilation: Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """
        Convolution with causal kernels over time

        Parameters
        ----------
        Same parameters as Conv2d
        """
        super().__init__()
        self.causal_pad_amount = self._get_causal_pad_amount(kernel_size, stride, dilation)

        # error handling
        err_msg = "only stride[0] == 1 is supported"
        assert get_time_value(stride) == 1, err_msg
        err_msg = "temporal padding cannot be set explicitely"
        assert get_time_value(padding) == 0, err_msg

        # inner modules
        self.causal_pad = nn.ConstantPad2d((0, 0, self.causal_pad_amount, 0), 0)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

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
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: nn.Module = nn.ReLU(),
        residual_merge: Optional[Callable] = None,
        device=None,
        dtype=None,
    ) -> None:
        """
        CausalConv2dNormAct + BatchNorm2d + Activation.

        Parameters
        ----------
        Combination of the modules parameters

        activation: nn.Module, optional
            Activation module, by default nn.Relu()
        residual_merge: Optional[Callable], optional
            If different da None, it indicates the merge operation after
            the activation, by default None
        """
        super().__init__()

        # inner modules
        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
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
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y
