from torchaudio.functional import resample
from typing import Tuple, Type, Union
import torch.nn.functional as F
from torch import Tensor, nn
from pathlib import Path
import soundfile as sf
import numpy as np
import torchaudio
import torch
import yaml

# export list
__all__ = [
    "DotDict",
    "Config",
    "Config",
    "load_audio",
    "save_audio",
    "stft",
    "istft",
    "get_device",
    "to_numpy",
]

# = = = = generic utilities


class DotDict(dict):
    """
    A dict that allows dot notation
    for accessing to the elements
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config:
    def __init__(self, config_path: Path) -> None:
        """
        A class to handle yaml configuration files.
        Inside a configuration many sections can be defined,
        with parameters within them.

        Parameters
        ----------
        config_path : Path
            Path to the yaml configuration file
        """
        # loading the configuration
        err_msg = "only YAML configurations are supported"
        assert config_path.suffix == ".yaml", err_msg
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, section: str, parameter: str, type: Type = str, default=None):
        """
        Gets a parameter from the configuration.

        Parameters
        ----------
        section : str
            Configuration section
        parameter : str
            Name of the parameter
        type : Type, optional
            Type of the parameter, by default str
        default : optional
            Default if the parameter does not exist, by default None

        Returns
        -------
        Fetched parameter
        """
        cfg = self.config

        try:
            # getting the section
            sec = cfg[section]
            # getting the parameter
            param = type(sec[parameter])
        except KeyError:
            return default

        return param


# = = = = io utilities


def load_audio(
    file_path: Path,
    sample_rate: int = None,
    tensor: bool = False,
) -> Tuple[Union[np.ndarray, Tensor], int]:
    """
    Loads an audio file.

    Parameters
    ----------
    file_path : Path
        Path to the audio file
    sample_rate : int, optional
        Target sample rate, by default None
    tensor : bool, optional
        If True loads a torch Tensor, by default False

    Returns
    -------
    Tuple[np.ndarray, int]
        audio, sample_rate
    """
    if not tensor:
        data, sample_rate = sf.read(file_path, samplerate=sample_rate)
        data = data.T
        if len(data.shape) == 1:
            data = data[None, :]
    else:
        data, old_sample_rate = torchaudio.load(file_path)
        if sample_rate is None:
            sample_rate = old_sample_rate
        else:
            data = resample(data, old_sample_rate, sample_rate)

    return data, sample_rate


def save_audio(file_path: Path, data: Union[np.ndarray, Tensor], sample_rate: int):
    """
    Saves an audio file

    Parameters
    ----------
    file_path : Path
        Path to the audio file
    data : Union[np.ndarray, Tensor]
        Audio file to save
    sample_rate : int, optional
        Target sample rate, by default None
    """
    dtype = type(data)

    if dtype == np.ndarray:
        sf.write(file_path, data.T, samplerate=sample_rate)
    elif dtype == Tensor:
        torchaudio.save(file_path, data, sample_rate=sample_rate)
    else:
        err_msg = f"{dtype} is not supported by save_audio"
        raise NotImplementedError(err_msg)


# = = = = audio utilities


def stft(
    x: Union[np.ndarray, Tensor],
    sample_rate: int = 16000,
    framesize_ms: int = 10,
    window="hann_window",
    window_overlap=0.5,
    frame_oversampling=4,
) -> Union[np.ndarray, Tensor]:
    """
    Calculates the STFT of a signal.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal of shape (..., T)
    sample_rate : int, optional
        Sample rate of the signal, by default 16000
    framesize_ms : int, optional
        STFT framesize in ms, by default 10
    window : str, optional
        Torch window to use, by default "hann_window"
    window_overlap : float, optional
        Window overlap factor between frames, by default 0.5
    frame_oversampling : int, optional
        Lef zero padding applied for each frame (1 equals to no zero pad), by default 4

    Returns
    -------
    Union[np.ndarray, Tensor]
        STFT of the input of shape (..., T', F')

    Raises
    ------
    AttributeError
        If the window chosen does not exist
    """
    return _stft_istft_core(
        x, True, sample_rate, framesize_ms, window, window_overlap, frame_oversampling
    )


def istft(
    x: Union[np.ndarray, Tensor],
    sample_rate: int = 16000,
    framesize_ms: int = 10,
    window="hann_window",
    window_overlap=0.5,
    frame_oversampling=4,
) -> Union[np.ndarray, Tensor]:
    """
    Calculates the ISTFT of a signal.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal of shape (..., T, F)
    sample_rate : int, optional
        Sample rate of the signal, by default 16000
    framesize_ms : int, optional
        STFT framesize in ms, by default 10
    window : str, optional
        Torch window to use, by default "hann_window"
    window_overlap : float, optional
        Window overlap factor between frames, by default 0.5
    frame_oversampling : int, optional
        Lef zero padding applied for each frame (1 equals to no zero pad), by default 4

    Returns
    -------
    Union[np.ndarray, Tensor]
        ISTFT of the input of shape (..., T')

    Raises
    ------
    AttributeError
        If the window chosen does not exist
    """
    return _stft_istft_core(
        x, False, sample_rate, framesize_ms, window, window_overlap, frame_oversampling
    )


def _stft_istft_core(
    x: Union[np.ndarray, Tensor],
    is_stft: bool,
    sample_rate: int = 16000,
    framesize_ms: int = 10,
    window="hann_window",
    window_overlap=0.5,
    frame_oversampling=2,
) -> Union[np.ndarray, Tensor]:
    """
    Calculates the STFT/ISTFT of a signal.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal
    sample_rate : int, optional
        Sample rate of the signal, by default 16000
    framesize_ms : int, optional
        STFT framesize in ms, by default 10
    window : str, optional
        Torch window to use, by default "hann_window"
    window_overlap : float, optional
        Window overlap factor between frames, by default 0.5
    frame_oversampling : int, optional
        Lef zero padding applied for each frame (1 equals to no zero pad), by default 4

    Returns
    -------
    Union[np.ndarray, Tensor]
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
        win_fun = getattr(torch, window)
    except AttributeError:
        allowed_win = [
            "hann_window",
            "hamming_window",
            "bartlett_window",
            "blackman_window",
            "kaiser_window",
        ]
        err_msg = "choose a window between:\n" + ", ".join(allowed_win)
        raise AttributeError(err_msg)

    # parameters of the STFT
    win_length = int(sample_rate * framesize_ms / 1000)
    hop_size = int(win_length * window_overlap)
    n_fft = int(win_length * frame_oversampling)
    _window = torch.zeros(n_fft)
    _window[:win_length] = win_fun(win_length)

    # STFT/ISTFT dependent code
    _transpose = lambda x: x.transpose(-1, -2)
    if is_stft:
        transform = torch.stft
        # compensating for oversampling
        padding = n_fft - win_length
        x = F.pad(x, (0, padding))
    else:
        transform = torch.istft
        x = _transpose(x)
        # fix for torch NOLA check
        eps = 1e-5
        _window[_window < eps] = eps

    y = transform(
        x,
        n_fft=n_fft,
        hop_length=hop_size,
        window=_window,
        return_complex=is_stft,
        center=False,
    )

    # reshaping
    if is_stft:
        y = _transpose(y)

    if in_type == np.ndarray:
        # converting to numpy
        y = to_numpy(y)

    return y


# = = = = pytorch utilities


def get_device() -> str:
    """
    Gets the first CUDA device available or CPU
    if no CUDA device is available.

    Returns
    -------
    str
        Device id
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x: Tensor) -> np.ndarray:
    """
    Converts a Tensor into a numpy array.

    Parameters
    ----------
    x : Tensor
        Original tensor

    Returns
    -------
    np.ndarray
        Converted np array
    """
    return x.cpu().detach().numpy()
