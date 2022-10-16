from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pathimport import set_module_root

set_module_root(".", prefix=True)
from torch_utils.common import get_device

__all__ = ["Lookahead", "CausalConv2d"]


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
