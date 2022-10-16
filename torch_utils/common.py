from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler
from torchaudio.functional import resample
from typing import Tuple, Type, Union
import torch.nn.functional as F
from random import randrange
from pathlib import Path
from torch import Tensor
import soundfile as sf
import numpy as np
import torchaudio
import torch
import h5py
import yaml

# export list
__all__ = [
    # generic utilities
    "DotDict",
    "Config",
    "get_np_or_torch",
    # io utilities
    "load_audio",
    "save_audio",
    # audio utilities
    "stft",
    "istft",
    "db",
    "invert_db",
    "power",
    "energy",
    "rms",
    "snr",
    "fade_sides",
    "trim",
    # pytorch utilities
    "get_device",
    "to_numpy",
    # pytorch data loading
    "WeakShufflingSampler",
    "HDF5Dataset",
    "get_hdf5_dataloader",
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


def get_np_or_torch(x: Union[np.ndarray, Tensor]):
    """
    Returns numpy or torch modules depending on the input

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input

    Returns
    -------
    Module
        numpy or torch
    """
    if isinstance(x, Tensor):
        return torch
    else:
        torch.iscomplex = torch.is_complex
        torch.hanning = torch.hann_window
        return np


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
        data, old_sample_rate = sf.read(file_path)
        data = data.T
        if len(data.shape) == 1:
            data = data[None, :]
        if (sample_rate is not None) and (old_sample_rate != sample_rate):
            data = to_numpy(resample(Tensor(data), old_sample_rate, sample_rate))
    else:
        data, old_sample_rate = torchaudio.load(file_path)
        data = data.to(get_device())
        if sample_rate is None:
            sample_rate = old_sample_rate
        elif old_sample_rate != sample_rate:
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
        data = data.to("cpu")
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
        x = torch.from_numpy(x).to(get_device())

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
    _window = torch.zeros(n_fft).to(get_device())
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


def db(x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Converts linear to dB

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal amplitude

    Returns
    -------
    float
        Input in dB
    """
    eps = 1e-12
    module = get_np_or_torch(x)
    return 20 * module.log10(x + eps)


def invert_db(x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Converts dB to linear

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal amplitude in dB

    Returns
    -------
    float
        Input inverting dB
    """
    return 10 ** (x / 20)


def power(x: Union[np.ndarray, Tensor]) -> float:
    """
    Power of a signal, calculated for each channel.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal

    Returns
    -------
    float
        Power of the signal
    """
    module = get_np_or_torch(x)
    _power = module.einsum("...t,...t->...", x, x.conj())
    _power = module.einsum("...t,...t->...", x, x.conj())
    return _power


def energy(x: Union[np.ndarray, Tensor]) -> float:
    """
    Energy of a signal, calculated for each channel.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal

    Returns
    -------
    float
        Energy of the signal
    """
    samples = x.shape[-1]
    return power(x) / samples


def rms(x: Union[np.ndarray, Tensor]) -> float:
    """
    RMS of a signal, calculated for each channel.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input signal

    Returns
    -------
    float
        RMS of the signal
    """
    module = get_np_or_torch(x)
    return module.sqrt(energy(x))


def snr(x: Union[np.ndarray, Tensor], noise: Union[np.ndarray, Tensor]) -> float:
    """
    Signal to Noise Ratio (SNR) ratio in dB,
    calculated considering the RMS.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Signal of interest
    noise : Union[np.ndarray, Tensor]
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
    x: Union[np.ndarray, Tensor],
    win: Union[np.ndarray, Tensor],
    fade_len: int,
) -> Union[np.ndarray, Tensor]:
    """
    Handler used to apply a window over the sides of
    a signal.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input of shape [..., C, T, F]
    win : Union[np.ndarray, Tensor]
        Window
    fade_len : Union[np.ndarray, Tensor]
        Length of each fade in samples

    Returns
    -------
    Union[np.ndarray, Tensor]
        Faded output
    """
    x[..., :fade_len, :] *= win[:fade_len, None]
    x[..., -fade_len:, :] *= win[-fade_len:, None]
    return x


def fade_sides(x: Union[np.ndarray, Tensor], fade_len: int = 10) -> Union[np.ndarray, Tensor]:
    """
    Apply an half of an Hanning window to both
    sides of the input, in order to obtain a fade in/out.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input of shape [..., C, T, F]
    fade_len : int, optional
        Length of the fade in samples, by default 10.
        The length of the window is 2 * fade_len + 1.

    Returns
    -------
    Union[np.ndarray, Tensor]
        Faded output
    """
    module = get_np_or_torch(x)
    win = module.hanning(2 * fade_len + 1)
    if module == np:
        y = x.copy()
    else:
        win = win.to(get_device())
        win[-1] = 0
        y = x.detach().clone().to(get_device())
    y = _win_to_sides(y, win, fade_len)

    return y


def trim(
    x: Union[np.ndarray, Tensor],
    sample_rate: int,
    duration: float = 3,
) -> Union[np.ndarray, Tensor]:
    """
    Extracts a random temporal selection from the input.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input of shape [..., T, F]
    sample_rate : int
        Sample rate in Hz
    duration : float, optional
        Duration of the selection, by default 3 s

    Returns
    -------
    Union[np.ndarray, Tensor]
        Random temporal selection of the input
    """
    # calculating start and stop samples
    duration_samples = duration * sample_rate
    selection_start_max = x.shape[-2] - duration_samples
    start = randrange(0, selection_start_max)
    end = start + duration_samples

    # applying selection
    return x[..., start:end, :]


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


# = = = = pytorch data loading


class WeakShufflingSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        """
        Sampler that implements weak-shuffling.

        E.g. if dataset is [1,2,3,4] with batch_size=2,
             then first batch, [[1,2], [3,4]] then
             shuffle batches -> [[3,4],[1,2]]

        Referenece:
        https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc

        Parameters
        ----------
        dataset : Dataset
            Source dataset
        batch_size : int
            Size of the batches
        """
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = int(np.ceil((self.dataset_length / self.batch_size)))
        self.batch_ids = torch.randperm(self.n_batches)

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        datalen = self.dataset_length
        for id in self.batch_ids:
            start = id * self.batch_size
            end = (id + 1) * self.batch_size
            end = end if end < datalen else datalen
            selection = torch.arange(start, end)
            for index in selection:
                yield int(index)


class HDF5Dataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        data_layout: list,
    ) -> None:
        """
        Dataset supporting HDF5 format.

        The following constraints should be satisfied
        by the HDF5 file for a correct behaviour:
        - Multiple single-level groups (e.g. /group1, /group2, ...)
        - The batch size (first dimension) of each dataset is the same
        - Each group has len(data_layout) dataset inside

        Parameters
        ----------
        dataset_path : Path
            Path to the .hdf5 file
        data_layout : list
            Dictionary describing the layout of the data
            inside each group. For an input-label1-label2
            dataset the list would be ["input", "label1", "label2"]
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.data_layout = data_layout
        self.dataset_file = h5py.File(self.dataset_path, "r")
        self.groups = list(self.dataset_file.keys())
        self.group_batch_len = self.dataset_file[self.groups[0]][self.data_layout[0]].shape[0]
        self.data_layout = data_layout
        self._cache = None
        self._cache_idx = None

    def __len__(self):
        return len(self.groups) * self.group_batch_len

    def __getitem__(self, idx):
        # error handling
        err_msg = None
        if not isinstance(idx, list):
            err_msg = "HDF5Dataset must be sampled by a BatchLoader"
        elif len(idx) > self.group_batch_len:
            err_msg = "Reduce the batch size to be less than the batch size of the groups"
        elif self.group_batch_len % len(idx) != 0:
            err_msg = "Modify the batch size to be a divider of the HDF5 group batch size"
        if err_msg is not None:
            raise RuntimeError(err_msg)

        # cache update
        if not self._in_cache(idx[0]):
            self._update_cache(idx[0])

        # using the cache
        gbl = self.group_batch_len
        a, b = idx[0] % gbl, idx[-1] % gbl
        data = {k: d[a:b] for k, d in self._cache.items()}

        return data

    def _update_cache(self, idx: int) -> None:
        """
        Caches a group in memory.

        Parameters
        ----------
        idx : int
            Index of an element of the group
        """
        # getting the starting index of a group
        self._cache_idx = idx

        # updating the cache
        del self._cache
        g_idx = idx // self.group_batch_len
        g = self.dataset_file[self.groups[g_idx]]
        cast = lambda x: torch.from_numpy(np.array(x)).to(get_device())
        data = {k: cast(g[k]) for k in self.data_layout}
        self._cache = data

    def _in_cache(self, idx: int) -> bool:
        """
        Checks if an index is inside the cache.

        Parameters
        ----------
        idx : int
            Target index

        Returns
        -------
        bool
            True if the element is inside the cache
        """
        c_idx = self._cache_idx
        flag = (c_idx is not None) and (idx >= c_idx and idx < (c_idx + self.group_batch_len))
        return flag


def get_hdf5_dataloader(
    dataset: HDF5Dataset,
    batch_size: int = None,
    dataloader_kwargs: dict = None,
):
    """
    Create a dataloader binded to a HDF5Dataset.

    Parameters
    ----------
    dataset : HDF5Dataset
        HDF5 Dataset
    batch_size : int, optional
        Batch size of the dataloader, by default HDF5Dataset.group_batch_len
    dataloader_kwargs : dict, optional
        DataLoader arguments, by default {}
    """
    if batch_size is None:
        batch_size = dataset.group_batch_len

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    sampler = BatchSampler(
        SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        **dataloader_kwargs,
    )

    return dataloader
