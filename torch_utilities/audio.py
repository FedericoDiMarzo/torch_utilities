__all__ = [
    "stft",
    "istft",
    "MelFilterbank",
    "MelInverseFilterbank",
    "db",
    "invert_db",
    "power",
    "energy",
    "rms",
    "fade_sides",
    "random_trim",
    "trim_silence",
    "interleave",
    "trim_as_shortest",
]

from torchaudio.functional import melscale_fbanks
from typing import Optional, List
from random import randrange
from torch import Tensor
import numpy as np
import torch


from torch_utilities.common import (
    get_np_or_torch,
    TensorOrArray,
    transpose,
    to_numpy,
)


# TODO: Use asteroid_filterbanks
def stft(
    x: TensorOrArray,
    sample_rate: int = 16000,
    hopsize_ms: int = 10,
    window: str = "hann",
    win_len_ms: int = 20,
    win_oversamp: int = 2,
) -> TensorOrArray:
    """
    Calculates the STFT of a signal.

    Parameters
    ----------
    x : TensorOrArray
        Input signal of shape (..., T)
    sample_rate : int, optional
        Sample rate of the signal, by default 16000
    hopsize_ms : int, optional
        STFT hopsize in ms, by default 10
    window : str, optional
        Torch window to use, by default "hann"
    win_len_ms : int, optional
        Window length in ms, by default 20 ms
    win_oversamp : int, optional
        Zero padding applied equal to the window length
        (1 equals to no zero pad), by default 2

    Returns
    -------
    TensorOrArray
        STFT of the input of shape (..., T', F')

    Raises
    ------
    AttributeError
        If the window chosen does not exist
    """
    raise NotImplementedError()


def istft(
    x: TensorOrArray,
    sample_rate: int = 16000,
    hopsize_ms: int = 10,
    window: str = "hann",
    win_len_ms: int = 20,
    win_oversamp: int = 2,
) -> TensorOrArray:
    """
    Calculates the ISTFT of a signal.

    Parameters
    ----------
    x : TensorOrArray
        Input signal of shape (..., T, F)
    sample_rate : int, optional
        Sample rate of the signal, by default 16000
    hopsize_ms : int, optional
        STFT hopsize in ms, by default 10
    window : str, optional
        Torch window to use, by default "hann"
    win_len_ms : int, optional
        Window length in ms, by default 20 ms
    win_oversamp : int, optional
        Zero padding applied equal to the window length
        (1 equals to no zero pad), by default 2

    Returns
    -------
    TensorOrArray
        ISTFT of the input of shape (..., T')

    Raises
    ------
    AttributeError
        If the window chosen does not exist
    """
    raise NotImplementedError()


class MelFilterbank:
    def __init__(
        self,
        sample_rate: int,
        n_freqs: int,
        n_mels: int,
        device: Optional[torch.device] = None,
    ):
        """
        Apply Mel filterbank to the input batch.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the signal
        n_freqs : int
            Stft frequencies
        n_mels : int
            Number of mel frequencies
        device : Optional[torch.device], optional
            Device for the filterbank matrix, by default None
        """
        self.sample_rate = sample_rate
        self.n_freqs = n_freqs
        self.n_mels = n_mels
        self.device = device
        self.filterbank = self._get_filterbank()

    def _get_filterbank(self) -> Tensor:
        """
        Gets the mel filterbank

        Returns
        -------
        Tensor
            Mel filterbank matrix
        """
        filterbank = melscale_fbanks(
            n_freqs=self.n_freqs,
            n_mels=self.n_mels,
            f_min=30,
            f_max=self.sample_rate // 2,
            sample_rate=self.sample_rate,
        )
        filterbank = filterbank.to(self.device)

        return filterbank

    def __call__(self, x: TensorOrArray) -> Tensor:
        """
        Parameters
        ----------
        x : TensorOrArray
            Signal of shape (B, C, T, n_freq)

        Returns
        -------
        Tensor
            STFT of shape (B, C, T, n_mel)
        """
        is_np = get_np_or_torch(x) == np
        if is_np:
            x = torch.from_numpy(x).to(self.device)

        # handling dtypes
        x = x.abs()
        self.filterbank = self.filterbank.to(x.dtype)

        y = x @ self.filterbank

        if is_np:
            y = to_numpy(y)
        return y


class MelInverseFilterbank:
    def __init__(
        self,
        sample_rate: int,
        n_freqs: int,
        n_mels: int,
        device: Optional[torch.device] = None,
    ):
        """
        Apply inverse Mel filterbank to the input batch,
        to get back a spectrogram.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the signal
        n_freqs : int
            Stft frequencies
        n_mels : int
            Number of mel frequencies
        device : Optional[torch.device], optional
            Device for the filterbank matrix, by default None
        """
        self.sample_rate = sample_rate
        self.n_freqs = n_freqs
        self.n_mels = n_mels
        self.device = device
        self.filterbank = self._get_filterbank()

    def _get_filterbank(self) -> Tensor:
        """
        Gets the mel filterbank

        Returns
        -------
        Tensor
            Mel filterbank matrix
        """
        filterbank = melscale_fbanks(
            n_freqs=self.n_freqs,
            n_mels=self.n_mels,
            f_min=30,
            f_max=self.sample_rate // 2,
            sample_rate=self.sample_rate,
        )
        # pseudo-inverse is used to approximate
        # the inverse transform
        filterbank = torch.linalg.pinv(filterbank)
        filterbank = filterbank.to(self.device)

        return filterbank

    def __call__(self, x: TensorOrArray) -> TensorOrArray:
        """
        Parameters
        ----------
        x : TensorOrArray
            Signal of shape (B, C, T, n_freq)

        Returns
        -------
        TensorOrArray
            STFT of shape (B, C, T, n_mel)
        """
        is_np = get_np_or_torch(x) == np
        if is_np:
            x = torch.from_numpy(x).to(self.device)

        # handling dtypes
        x = x.abs()
        self.filterbank = self.filterbank.to(x.dtype)

        y = x @ self.filterbank

        if is_np:
            y = to_numpy(y)
        return y


def db(x: TensorOrArray, eps: float = 1e-12) -> TensorOrArray:
    """
    Converts linear to dB

    Parameters
    ----------
    x : TensorOrArray
        Input signal amplitude
    eps : float
        Number summed to x before applying the logarithm,
        by default 1e-12

    Returns
    -------
    float
        Input in dB
    """
    module = get_np_or_torch(x)
    return 20 * module.log10(x + eps)


def invert_db(x: TensorOrArray, eps: float = 1e-12) -> TensorOrArray:
    """
    Converts dB to linear

    Parameters
    ----------
    x : TensorOrArray
        Input signal amplitude in dB

    Returns
    -------
    float
        Input inverting dB
    """
    return 10 ** (x / 20) - eps


def power(x: TensorOrArray) -> float:
    """
    Power of a signal, calculated for each channel.

    Parameters
    ----------
    x : TensorOrArray
        Input signal of shape (..., T)

    Returns
    -------
    float
        Power of the signal of shape (...) (len(x) - 1)
    """
    module = get_np_or_torch(x)
    pwr = module.einsum("...t,...t->...", x, x.conj())
    if module == torch:
        pwr = pwr.item()
    return pwr


def energy(x: TensorOrArray) -> float:
    """
    Energy of a signal.

    Parameters
    ----------
    x : TensorOrArray
        Input signal of shape (..., T)

    Returns
    -------
    float
        Energy of the signal of shape (...) (len(x) - 1)
    """
    samples = x.shape[-1]
    return power(x) / samples


def rms(x: TensorOrArray) -> float:
    """
    RMS of a signal, calculated for each channel.

    Parameters
    ----------
    x : TensorOrArray
        Input signal of shape (..., T)

    Returns
    -------
    float
        RMS of the signal of shape (...) (len(x) - 1)
    """
    e = energy(x)
    return np.sqrt(e)


def _win_to_sides(
    x: TensorOrArray,
    win: TensorOrArray,
    fade_len: int,
    direction: str,
) -> TensorOrArray:
    """
    Handler used to apply a window over the sides of
    a signal.

    Parameters
    ----------
    x : TensorOrArray
        Input of shape (..., C, T)
    win : TensorOrArray
        Window
    fade_len : TensorOrArray
        Length of each fade in samples
    direction : str
        Indicates the sides in which the fade is applied,
        one between "left", "right" or "both"

    Returns
    -------
    TensorOrArray
        Faded output
    """
    # error handling
    err_msg = 'direction must be one between "left", "right" or "both"'
    assert direction in ["left", "right", "both"], err_msg

    if direction in ["left", "both"]:
        x[..., :fade_len] *= win[:fade_len]
    if direction in ["right", "both"]:
        x[..., -fade_len:] *= win[-fade_len:]
    return x


def fade_sides(
    x: TensorOrArray, fade_len: int = 100, direction: str = "both"
) -> TensorOrArray:
    """
    Apply an half of an Hanning window to the
    sides of the input, in order to obtain a fade in/out.

    Parameters
    ----------
    x : TensorOrArray
        Input of shape (..., C, T)
    fade_len : int, optional
        Length of the fade in samples, by default 10.
        The length of the window is 2 * fade_len + 1.
    direction : str
        Indicates the sides in which the fade is applied,
        one between "left", "right" or "both", by default "both"

    Returns
    -------
    TensorOrArray
        Faded output
    """
    module = get_np_or_torch(x)
    win = module.hanning(2 * fade_len + 1)
    if module == np:
        y = x.copy()
    else:
        win = win.to(x.device)
        win[-1] = 0
        y = x.detach().clone()
    y = _win_to_sides(y, win, fade_len, direction)

    return y


def random_trim(
    x: TensorOrArray,
    sample_rate: int,
    duration: float = 3,
) -> TensorOrArray:
    """
    Extracts a random temporal selection from the input.

    Parameters
    ----------
    x : TensorOrArray
        Input of shape (..., T)
    sample_rate : int
        Sample rate in Hz
    duration : float, optional
        Duration of the selection, by default 3 s

    Returns
    -------
    TensorOrArray
        Random temporal selection of the input
    """
    module = get_np_or_torch(x)
    x_len = x.shape[-1]
    duration_samples = int(duration * sample_rate)
    selection_start_max = x_len - duration_samples

    if selection_start_max <= 0:
        # if the input is longer than the selection
        # just zero pad the input
        y = module.zeros((*x.shape[:-1], duration_samples))
        y[..., :x_len] = x
    else:
        # applying selection
        start = randrange(0, selection_start_max)
        end = start + duration_samples
        y = x[..., start:end]
    return y


def trim_silence(
    x: TensorOrArray,
    threshold_db: float = -35,
    margin: int = 0,
) -> TensorOrArray:
    """
    Trims the silences at the beginning and end of a sample.

    Parameters
    ----------
    x : TensorOrArray
        Input sample of shape (T,)
    threshold_db : float, optional
        Relative to x.max() to detect the silences, by default -35 dB
    margin : int, optional
        Samples kept at both sides after the trimming, by default 0

    Returns
    -------
    TensorOrArray
        Trimmed ouput of shape (T',)
    """
    module = get_np_or_torch(x)

    # finding the start and end points
    threshold = invert_db(threshold_db)
    threshold *= module.abs(x).max()
    thr = module.zeros_like(x, dtype=int)
    thr[module.abs(x) > threshold] = 1
    thr = thr if module == np else thr.cpu().detach().numpy()
    thr = thr.tolist()

    try:
        start = thr.index(1)
        end = len(thr) - thr[::-1].index(1)

        # trimming the silences
        x_start = int(np.clip(start - margin, 0, None))
        x_end = end + margin
        y = x[..., x_start:x_end]
        return y
    except ValueError:
        # no value found
        return x


def interleave(*xs: List[TensorOrArray]) -> Tensor:
    """
    Interleaves many input tensors in one over the last dimension.

    Parameters
    ----------
    xs : List[TensorOrArray]
        Input signals of the same shape

    Returns
    -------
    Tensor
        Interleaved signal, the shape is the same of the inputs except
        for the interleaved dimension of lenght D that will be len(xs)*D
    """
    mod = get_np_or_torch(xs[0])
    if mod == np:
        stride = len(xs)
        new_shape = list(xs[0].shape)
        new_shape[-1] *= stride
        y = mod.zeros(new_shape)

        for i, x in enumerate(xs):
            y[..., i::stride] = x
    else:
        y = torch.stack(xs, -1).flatten(-2, -1)

    return y


def trim_as_shortest(*xs: List[TensorOrArray], dim: int = -1) -> List[TensorOrArray]:
    """
    Trims all the inputs to the same length of the shortest one.

    Parameters
    ----------
    xs : List[TensorOrArray]
        Input signals of the same shape

    dim : int, optional
        Dimension along which the trimming is applied, by default -1

    Returns
    -------
    List[TensorOrArray]
        Trimmed signals
    """
    min_len = min([x.shape[dim] for x in xs])
    xs = [transpose(x, dim, -1) for x in xs]
    xs = [x[..., :min_len] for x in xs]
    xs = [transpose(x, dim, -1) for x in xs]
    return xs
