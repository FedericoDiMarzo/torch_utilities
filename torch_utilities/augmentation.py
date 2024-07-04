__all__ = [
    "shuffle",
    "dc_removal",
    "add_noise",
    "random_scaling",
    "random_overdrive",
    "random_lowpass",
    "random_highpass",
    "random_peak_eq",
]


# Augmentations support only Tensor, not numpy arrays

import torchaudio.functional as F
from torch import Tensor
from typing import Tuple
import numpy as np
import torch


from torch_utilities.audio import invert_db, rms


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

_rand_in_range = lambda a, b, n, dev: torch.rand(n, device=dev) * (b - a) + a

_1rand_in_range = lambda a, b: np.random.rand(1)[0] * (b - a) + a

_max_over_batch = lambda x: torch.max(x.flatten(1), dim=1)[0]

_normalize = lambda x: x / _max_over_batch(x + 1e-12)[:, None, None]

_expand2 = lambda x: x[:, None]

_expand3 = lambda x: x[:, None, None]

_safe_div = lambda n, d, eps: n / (d + eps)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


def dc_removal(x: Tensor) -> Tensor:
    """
    DC removal for each batch and channel.

    Parameters
    ----------
    x : Tensor
        Input signal of shape (B, C, T)

    Returns
    -------
    Tensor
        Processed output without DC removal of shape (B, C, T)
    """
    m = x.mean(dim=2)
    x -= m[..., None]
    return x


def add_noise(
    x: Tensor, n: Tensor, snr_range: Tuple[float, float]
) -> Tuple[Tensor, Tensor]:
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
    Tuple[Tensor]
        (
            noisy version of x of shape (B, C, T),
            per-batch SNR of shape (B,)
    """
    a, b = _db_to_lin_range(snr_range)
    snr = _rand_in_range(a, b, x.shape[0], x.device)

    # the output will match these peaks
    x_peaks = _max_over_batch(x)

    # scaling the noise
    eps = 1e-8
    n = _safe_div(n, _expand2(rms(n)), eps)
    n *= _expand2(rms(x))
    n = _safe_div(n, _expand3(snr), eps)

    # summing and scaling back
    y = x + n
    y = _normalize(y)
    y *= _expand3(x_peaks)

    return y, snr


def random_scaling(x: Tensor, range_db: Tuple[float, float]) -> Tuple[Tensor, Tensor]:
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
    Tuple[Tensor, Tensor]
        (
            scaled version of x of shape (B, ...),
            per-batch scaling of shape (B,)
        )
    """
    a, b = _db_to_lin_range(range_db)
    scale = _rand_in_range(a, b, x.shape[0], x.device)
    x = _normalize(x)
    x *= _expand3(scale)
    return x, scale


def random_overdrive(
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
        Distorted version of x of shape (..., T)
    """

    x_peaks = _max_over_batch(x)

    gain = _1rand_in_range(*gain_range)
    color = _1rand_in_range(*colour_range)
    y = F.overdrive(x, gain, color)

    y = _normalize(y)
    y *= _expand3(x_peaks)

    return y


def random_lowpass(
    x: Tensor,
    sample_rate: int,
    cutoff_range: Tuple[float, float],
    q_range: Tuple[float, float],
) -> Tensor:
    """
    Apply a random lowpass filter to the input batch.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (..., T)
    sample_rate : int
        Sample rate of the input
    cutoff_range : Tuple[float, float]
        Ranges of possible cutoffs in Hz
    q_range : Tuple[float, float]
        Range of possible Q factors

    Returns
    -------
    Tensor
        Filtered output of shape (..., T)
    """
    cutoff = _1rand_in_range(*cutoff_range)
    q = _1rand_in_range(*q_range)
    y = F.lowpass_biquad(x, sample_rate, cutoff, q)
    return y


def random_highpass(
    x: Tensor,
    sample_rate: int,
    cutoff_range: Tuple[float, float],
    q_range: Tuple[float, float],
) -> Tensor:
    """
    Apply a random highpass filter to the input batch.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (..., T)
    sample_rate : int
        Sample rate of the input
    cutoff_range : Tuple[float, float]
        Ranges of possible cutoffs in Hz
    q_range : Tuple[float, float]
        Range of possible Q factors

    Returns
    -------
    Tensor
        Filtered output of shape (..., T)
    """
    cutoff = _1rand_in_range(*cutoff_range)
    q = _1rand_in_range(*q_range)
    y = F.highpass_biquad(x, sample_rate, cutoff, q)
    return y


def random_peak_eq(
    x: Tensor,
    sample_rate: int,
    cutoff_range: Tuple[float, float],
    gain_range: Tuple[float, float],
    q_range: Tuple[float, float],
) -> Tensor:
    """
    Apply a random peak eq filter to the input batch.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (..., T)
    sample_rate : int
        Sample rate of the input
    cutoff_range : Tuple[float, float]
        Ranges of possible cutoffs in Hz
    q_range : Tuple[float, float]
        Range of possible Q factors

    Returns
    -------
    Tensor
        Filtered output of shape (..., T)
    """
    cutoff = _1rand_in_range(*cutoff_range)
    q = _1rand_in_range(*q_range)
    gain = _1rand_in_range(*gain_range)
    y = F.equalizer_biquad(x, sample_rate, cutoff, gain, q)
    return y
