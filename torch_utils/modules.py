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

    def forward(self, *x):
        return self.f(*x)


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

        # attributes
        self.lookahead = lookahead
        self.maintain_shape = maintain_shape

        # inner modules
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

        # attributes
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
        padding_f: Optional[int] = None,
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

        padding_f : Optional[int]
            Symmetric padding over frequency, by default keep the same frequency shape
            as the input if stride_f==1 else set padding_f=1
        separable : bool, optional
            Enable separable convolution (depthwise + pointwise), by default False
        enable_weight_norm : bool, optional
            Enables weight normalization, by default False
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
        self.enable_weight_norm = enable_weight_norm
        self.dtype = dtype

        # optional weight normalization
        self._normalize = weight_norm if self.enable_weight_norm else (lambda x: x)

        # inner modules
        self.causal_pad_amount = self._get_causal_pad_amount()
        self.groups = np.gcd(self.in_channels, self.out_channels)

        layers = []

        # causal padding
        causal_pad = nn.ConstantPad2d((0, 0, self.causal_pad_amount, 0), 0)
        layers.append(causal_pad)

        # convolution
        conv_pad = 0 if self.padding_f is None else (0, self.padding_f)
        if not self.separable:
            full_conv = self._normalize(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=(1, stride_f),
                    dilation=self.dilation,
                    bias=self.bias,
                    dtype=self.dtype,
                    padding=conv_pad,
                )
            )
            layers.append(full_conv)
        else:
            # separable convolution
            # depthwise + pointwise
            depthwise = self._normalize(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=(1, stride_f),
                    dilation=self.dilation,
                    bias=self.bias,
                    groups=self.groups,
                    dtype=self.dtype,
                    padding=conv_pad,
                )
            )
            pointwise = self._normalize(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=False,
                    dtype=self.dtype,
                )
            )
            layers += [depthwise, pointwise]

        # freq padding
        freq_pad = self._get_default_freq_padding()
        layers.append(freq_pad)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x

    def _get_causal_pad_amount(self) -> int:
        """
        Calculates the causal padding.
        """
        kernel_size_t, dilation_t = map(get_time_value, (self.kernel_size, self.dilation))
        causal_pad = get_causal_conv_padding(kernel_size_t, dilation_t)
        return causal_pad

    def _get_default_freq_padding(self) -> nn.Module:
        """
        Gets the default frequency padding.

        Returns
        -------
        nn.Module
            Frequency padding module
        """
        kernel_size_f, dilation_f = map(get_freq_value, (self.kernel_size, self.dilation))
        pad_f = dilation_f * (kernel_size_f - 1) + 1
        half_pad_f = pad_f // 2
        if self.padding_f is not None or self.stride_f != 1:
            pad = nn.Identity()  # manual padding
        elif pad_f % 2 == 0:
            pad = nn.ConstantPad2d((half_pad_f, half_pad_f - 1, 0, 0), 0)
        else:
            pad = nn.ConstantPad2d((half_pad_f, half_pad_f, 0, 0), 0)
        return pad


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


        This layer ensures f_in = f_out // stride_f only if #TODO: change that with padding_f=None in Conv2D
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
            If different da None, it indicates the merge operation using a skip connection
            from the output of the transposed conv to the output of the activation, by default None
        disable_batchnorm : bool, optional
            Disable the BatchNorm2d layer, by default False
        enable_weight_norm : bool, optional
            Uses the weight normalization instead of the batch normalization,
            by default False
        dtype : _type_, optional
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
        self.disable_batchnorm = disable_batchnorm
        self.enable_weight_norm = enable_weight_norm
        self.dtype = dtype

        # weight normalization disables batch normalization
        if self.enable_weight_norm:
            self.disable_batchnorm = True

        # inner modules
        kernel_f = get_freq_value(self.kernel_size)
        dilation_f = get_freq_value(self.dilation)
        half_padding_f = self._get_freq_padding(kernel_f, dilation_f)

        self.conv = CausalConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding_f=half_padding_f,
            stride_f=stride_f,
            dilation=self.dilation,
            bias=False,
            separable=self.separable,
            enable_weight_norm=self.enable_weight_norm,
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
        x = self.conv(x)
        y = self.batchnorm(x)
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
        post_conv_dilation: Optional[Tuple[int, int]] = None,
        disable_dilation_f: bool = False,
        post_conv_count: int = 1,
        tconv_kernel_f: Optional[int] = None,
        tconv_stride_f: int = 2,
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
        post_conv_dilation : Optional[Tuple[int,int]]
            Dilation of the post convolutions. Can be also an int, or a list
            of ints/Tuple[int, int] of length post_conv_count;
            by default the dilation is equal to the kernel to the power of
            the post_conv layer index
        disable_dilation_f : bool
            If True dilation_f==1 for each post_conv_dilation setting,
            by default False
        post_conv_count : int, optional
            Number of post convolutions, by default 1
        tconv_kernel_f : Optional[int], optional
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
            Merge operation performed between the layer output and a residual skip connection
            from the output of the transposed conv to the output of the activation, by default None
        dtype : optional
            Module dtype, by default None
        """
        super().__init__()

        # attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.post_conv_kernel_size = post_conv_kernel_size
        self.post_conv_dilation = post_conv_dilation
        self.disable_dilation_f = disable_dilation_f
        self.post_conv_count = post_conv_count
        self.tconv_kernel_f = tconv_kernel_f
        self.tconv_stride_f = tconv_stride_f
        self.separable = separable
        self.batchnorm_eps = batchnorm_eps
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_affine = batchnorm_affine
        self.batchnorm_track_running_stats = batchnorm_track_running_stats
        self.disable_batchnorm = disable_batchnorm
        self.enable_weight_norm = enable_weight_norm
        self.activation = activation
        self.residual_merge = residual_merge
        self.dtype = dtype

        # optional weight normalization
        self._normalize = weight_norm if self.enable_weight_norm else (lambda x: x)

        # weight normalization disables batch normalization
        if self.enable_weight_norm:
            self.disable_batchnorm = True

        # tconv_kernel_f_size is set to twice tconv_stride_f
        # to reduce the reconstruction artifacts
        if self.tconv_kernel_f is None:
            self.tconv_kernel_f = 2 * self.tconv_stride_f

        # inner modules
        self.tconv = self._normalize(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, self.tconv_kernel_f),
                stride=(1, self.tconv_stride_f),
                bias=False,
                dtype=self.dtype,
            )
        )

        pad_f = self.tconv_stride_f - self.tconv_kernel_f
        half_pad_f = pad_f // 2
        if pad_f % 2 == 0:
            self.padding_f = nn.ConstantPad2d((half_pad_f, half_pad_f, 0, 0), 0)
        else:
            self.padding_f = nn.ConstantPad2d((half_pad_f, half_pad_f + 1, 0, 0), 0)

        self.conv = self._get_conv_layers()

        if disable_batchnorm:
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

        self.activation = activation or nn.Identity()
        self.residual_merge = residual_merge

    def forward(self, x: Tensor) -> Tensor:
        x = self.tconv(x)
        x = self.padding_f(x)
        y = self.conv(x)
        y = self.batchnorm(y)
        y = self.activation(y)
        if self.residual_merge is not None:
            y = self.residual_merge(x, y)
        return y

    def _get_default_dilation(self) -> List[Tuple[int, int]]:
        """
        The default dilation is an increasing power of the kernel
        # e.g: d_t_conv2 = k_t**2

        Returns
        -------
        List[Tuple[int, int]]
            default dilation
        """
        k_t, k_f = [f(self.post_conv_kernel_size) for f in (get_time_value, get_freq_value)]
        dilation = [(k_t**x, k_f**x) for x in range(self.post_conv_count)]
        if self.disable_dilation_f:
            dilation = [(d[0], 1) for d in dilation]
        return dilation

    def _get_conv_layers(self) -> nn.Sequential:
        """
        Gets the convolutional layers.

        Returns
        -------
        nn.Sequential
            Sequence of one or more CausalConv2d layers
        """

        # error handling ~ ~ ~ ~ ~ ~ ~ ~
        if self.post_conv_dilation is None:
            err_msg = f"kernel size should be an int or Tuple[int] when post_conv_dilation is None"
            assert type(self.post_conv_kernel_size) in (int, tuple), err_msg
        else:
            err_msg = f"post_conv_count == {self.post_conv_count} is not enforced"
            for p in self.post_conv_kernel_size, self.post_conv_dilation:
                if self.post_conv_count == 1:
                    assert (type(p) == tuple and len(p) == 2) or (type(p) == int), err_msg
                else:
                    assert type(p) != int and len(p) == self.post_conv_count, err_msg
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        # to solve an unique case ~ ~ ~
        if self.post_conv_dilation is None:
            self.post_conv_dilation = self._get_default_dilation()
            self.post_conv_kernel_size = [self.post_conv_kernel_size] * self.post_conv_count
        elif self.post_conv_count == 1:
            self.post_conv_dilation = [self.post_conv_dilation]
            self.post_conv_kernel_size = [self.post_conv_kernel_size]
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        conv = []
        for k, d in zip(self.post_conv_kernel_size, self.post_conv_dilation):
            conv.append(
                CausalConv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=k,
                    padding_f=None,
                    dilation=d,
                    bias=False,
                    separable=self.separable,
                    enable_weight_norm=self.enable_weight_norm,
                    dtype=self.dtype,
                )
            )

        conv = nn.Sequential(*conv)
        return conv


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


class DenseConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int],
        dilation: Optional[Tuple[int, int]] = None,
        disable_dilation_f: bool = False,
        depth: int = 3,
        final_stride: int = 1,
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
        Dense block from  Dense CNN With Self-Attention for Time-Domain Speech Enhancement
        paper (https://ieeexplore.ieee.org/document/9372863).

        Parameters
        ----------
        channels : int
            Number of input and output channels
        kernel_size : Tuple[int, int]
            Kernel size. Can be also an int, or a list
            of ints/Tuple[int, int] of length of depth
        dilation : Optional[Tuple[int,int]]
            Dilation. Can be also an int, or a list
            of ints/Tuple[int, int] of length depth;
            by default the dilation is equal to the kernel to the power of
            the post_conv layer index
        disable_dilation_f : bool
            If True dilation_f==1 for each conv dilation setting,
            by default False
        tconv_kernel_f : Optional[int], optional
            Frequncy kernel size of the transposed convolution,
            by default twice tconv_stride_f
        final_stride : int, optional
            Stride of the last convolution, by default 1
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
            Merge operation performed between the input  the last activation output,
            by default None
        dtype : optional
            Module dtype, by default None
        """
        super().__init__()

        # attributes
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.disable_dilation_f = disable_dilation_f
        self.depth = depth
        self.final_stride = final_stride
        self.separable = separable
        self.batchnorm_eps = batchnorm_eps
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_affine = batchnorm_affine
        self.batchnorm_track_running_stats = batchnorm_track_running_stats
        self.disable_batchnorm = disable_batchnorm
        self.enable_weight_norm = enable_weight_norm
        self.activation = activation
        self.residual_merge = residual_merge
        self.dtype = dtype
