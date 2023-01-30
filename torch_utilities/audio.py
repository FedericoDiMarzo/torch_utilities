from torchaudio.functional import melscale_fbanks
from pathimport import set_module_root
from typing import Optional, List
import torch.nn.functional as F
from random import randrange
from torch import Tensor
import numpy as np
import torch

set_module_root(".")
from torch_utilities.common import get_device, get_np_or_torch, to_numpy, TensorOrArray


# export list
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
    "snr",
    "fade_sides",
    "random_trim",
    "trim_silence",
    "interleave",
]


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
    return _stft_istft_core(
        True,
        x,
        sample_rate,
        hopsize_ms,
        window,
        win_len_ms,
        win_oversamp,
    )


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
    return _stft_istft_core(
        False,
        x,
        sample_rate,
        hopsize_ms,
        window,
        win_len_ms,
        win_oversamp,
    )


def _stft_istft_core(
    is_stft: bool,
    x: TensorOrArray,
    sample_rate: int = 16000,
    hopsize_ms: int = 10,
    window: str = "hann",
    win_len_ms: int = 20,
    win_oversamp: int = 2,
) -> TensorOrArray:
    """
    Calculates the STFT/ISTFT of a signal.

    Parameters
    ----------
    is_stft : bool
        Selects between STFT and ISTFT
    x : TensorOrArray
        Input signal
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
        STFT of the input

    Raises
    ------
    AttributeError
        If the window chosen does not exist
    """
    # converting to Tensor
    in_type = type(x)
    if in_type == np.ndarray:
        x = torch.from_numpy(x)

    # getting the window function
    try:
        window += "_window"
        win_fun = getattr(torch, window)
    except AttributeError:
        allowed_win = [w + "_window" for w in ["hann", "hamming", "bartlett", "blackman", "kaiser"]]
        err_msg = "choose a window between:\n" + ", ".join(allowed_win)
        raise AttributeError(err_msg)

    # parameters of the STFT
    _ms_to_samples = lambda x: int(x * sample_rate / 1000)
    win_len = _ms_to_samples(win_len_ms)
    hopsize = _ms_to_samples(hopsize_ms)
    n_fft = int(win_len * win_oversamp)
    _window = torch.zeros(n_fft, device=x.device)
    _window[:win_len] = win_fun(win_len, device=x.device)

    # STFT/ISTFT dependent code
    _transpose = lambda x: x.transpose(-1, -2)
    if is_stft:
        transform = torch.stft
        # compensating for oversampling and center==True
        pad_ovr = n_fft - win_len
        pad_ctr = win_len // 2
        x = F.pad(x, (pad_ctr, pad_ovr))
    else:
        transform = torch.istft
        x = _transpose(x)

    y = transform(
        x,
        n_fft=n_fft,
        hop_length=hopsize,
        window=_window,
        return_complex=is_stft,
        center=True,
    )

    if is_stft:
        # reshaping
        y = _transpose(y)
        # compensating for center==True
        y = y[:, 1:]

    if in_type == np.ndarray:
        # converting to numpy
        y = to_numpy(y)

    return y


class MelFilterbank:
    def __init__(
        self,
        sample_rate: int,
        n_freqs: int,
        n_mels: int,
        device: Optional[torch.device] = None,
    ) -> None:
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
        self.device = device or get_device()
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
    ) -> None:
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
        self.device = device or get_device()
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
    _power = module.einsum("...t,...t->...", x, x.conj())
    return _power


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
    module = get_np_or_torch(x)
    return module.sqrt(energy(x))


def snr(x: TensorOrArray, noise: TensorOrArray) -> float:
    """
    Signal to Noise Ratio (SNR) ratio in dB,
    calculated considering the RMS.

    Parameters
    ----------
    x : TensorOrArray
        Signal of interest
    noise : TensorOrArray
        Interference

    Returns
    -------
    float
        SNR in db
    """
    err_msg0 = "snr supports only 1D and 2D signals"
    assert len(x.shape) in [1, 2], err_msg0
    assert len(noise.shape) in [1, 2], err_msg0
    err_msg1 = "x and noise should be of the same type"
    assert type(x) == type(noise), err_msg1

    module = get_np_or_torch(x)
    channel_mean = lambda x: module.mean(x, -2) if len(x.shape) == 1 else x
    a = channel_mean(db(rms(x)))
    b = channel_mean(db(rms(noise)))
    snr = a - b
    return snr


def _win_to_sides(
    x: TensorOrArray,
    win: TensorOrArray,
    fade_len: int,
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

    Returns
    -------
    TensorOrArray
        Faded output
    """
    x[..., :fade_len] *= win[:fade_len]
    x[..., -fade_len:] *= win[-fade_len:]
    return x


def fade_sides(x: TensorOrArray, fade_len: int = 100) -> TensorOrArray:
    """
    Apply an half of an Hanning window to both
    sides of the input, in order to obtain a fade in/out.

    Parameters
    ----------
    x : TensorOrArray
        Input of shape (..., C, T)
    fade_len : int, optional
        Length of the fade in samples, by default 10.
        The length of the window is 2 * fade_len + 1.

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
        win = win.to(get_device())
        win[-1] = 0
        y = x.detach().clone()
    y = _win_to_sides(y, win, fade_len)

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


# TODO: tests
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
    stride = len(xs)
    new_shape = list(xs[0].shape)
    new_shape[-1] *= stride
    y = mod.zeros(new_shape)

    for i, x in enumerate(xs):
        y[..., i:stride:] = x

    return y
