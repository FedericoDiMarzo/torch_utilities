from pathimport import set_module_root
from typing import Optional, Tuple, Union
import torchaudio.functional as F
from random import randrange
from torch import Tensor
import numpy as np
import torch


set_module_root(".", prefix=True)
from torch_utils.common import get_device
from torch_utils.audio import invert_db


# export list
__all__ = [
    "shuffle",
    "add_noise",
    "scale",
    "overdrive",
    "biquad",
]


def shuffle(x: Tensor) -> Tensor:
    """
    Shuffles a tensor over the batches.

    Parameters
    ----------
    x : Tensor
        Target tensor of shape (B, ...)

    Returns
    -------
    Tensor
        Shuffled tensor, same shape of x
    """
    y = x[torch.randperm(x.shape[0])]
    return y


# L-a-m-b-d-a-s = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

_db_to_lin_range = lambda xs: [invert_db(x) for x in xs]

_rand_in_range = lambda a, b, n: torch.rand(n) * (b - a) + a

_1rand_in_range = lambda a, b: np.random.rand(1)[0] * (b - a) + a

_max_over_batch = lambda x: torch.max(x.flatten(1), dim=1)[0]

_normalize = lambda x: x / _max_over_batch(x + 1e-12)

_expand3 = lambda x: x[:, None, None]


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


def add_noise(x: Tensor, n: Tensor, snr_range: Tuple[float, float]) -> Tensor:
    """
    Add noise at various SNRs.

    Parameters
    ----------
    x : Tensor
        Target tensor of shape (B, C, T)
    n : Tensor
        Noise tensor of shape (B, C, T)
    snr_range : Tuple[float, float]
        Max and min SNR

    Returns
    -------
    Tensor
        Noisy version of x, same shape of x
    """
    a, b = _db_to_lin_range(snr_range)
    snr = _rand_in_range(a, b, x.shape[0])

    # the output will match these peaks
    x_peaks = _max_over_batch(x)

    # scaling the noise
    _f = lambda z: z[:, None, None]
    n *= _expand3(x_peaks)
    n *= _expand3(snr)

    # summing and scaling back
    y = x + n
    y = _normalize(y)
    y *= x_peaks

    return y


def scale(x: Tensor, range_db: Tuple[float, float]) -> Tensor:
    """
    Applies a random scaling over the batches

    Parameters
    ----------
    x : Tensor
        Target tensor of shape (B, ...)
    range_db : Tuple[float, float]
        Range of the scaling in dB

    Returns
    -------
    Tensor
        Scaled version of x, same shape of x
    """
    a, b = _db_to_lin_range(range_db)
    scale = _rand_in_range(a, b, x.shape[0])
    x = _normalize(x)
    x *= scale
    return x


def overdrive(
    x: Tensor,
    gain_range: Tuple[float, float] = (60, 100),
    colour_range: Tuple[float, float] = (0, 100),
) -> Tensor:
    """
    Applies an overdrive to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (..., T)
    gain_range : Tuple[float, float], optional
        Range of the gain parameter, by default (60, 100)
    colour_range : Tuple[float, float], optional
        Range of the color parameter, by default (0, 100)

    Returns
    -------
    Tensor
        Distorted version of x, same shape of x
    """

    x_peaks = _max_over_batch(x)

    gain = _1rand_in_range(*gain_range)
    color = _1rand_in_range(*colour_range)
    y = F.overdrive(x, gain, color)

    y = _normalize(y)
    y *= _expand3(x_peaks)

    return y


# TODO: biquad
def biquad(x: Tensor) -> Tensor:
    raise NotImplementedError()
