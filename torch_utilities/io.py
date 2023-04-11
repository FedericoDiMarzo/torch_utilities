from typing import Iterator, List, Tuple, Union
from resampy import resample as resample_np
from torchaudio.functional import resample
from pathimport import set_module_root
from multiprocess import Pool
from itertools import islice
from pathlib import Path
from torch import Tensor
import soundfile as sf
import numpy as np
import torchaudio
import torch

set_module_root(".")
from torch_utilities.common import TensorOrArray
from torch_utilities.pytorch import get_device

# export list
__all__ = [
    "load_audio",
    "save_audio",
    "load_audio_parallel",
    "load_audio_parallel_itr",
]


def load_audio(
    file_path: Path,
    sample_rate: int = None,
    tensor: bool = False,
    device: str = "auto",
) -> Tuple[TensorOrArray, int]:
    """
    Loads an audio file.

    Parameters
    ----------
    file_path : Path
        Path to the audio file
    sample_rate : int, optional
        Target sample rate, by default avoids resample
    tensor : bool, optional
        If True loads a torch Tensor, by default False
    device : str, optional
        The device to load the tensor into, by default auto

    Returns
    -------
    Tuple[np.ndarray, int]
        (audio of shape (C, T), sample_rate)
    """
    # audio loading
    if not tensor:
        data, old_sample_rate = sf.read(
            file_path,
        )
        data = data.T
        if len(data.shape) == 1:
            data = data[None, :]
    else:
        data, old_sample_rate = torchaudio.load(file_path)
        data = data.to(get_device() if device == "auto" else device)

    # resampling
    if sample_rate is None:
        sample_rate = old_sample_rate
    elif old_sample_rate != sample_rate:
        if tensor:
            data = resample(data, old_sample_rate, sample_rate)
        else:
            data = resample_np(data, old_sample_rate, sample_rate)

    return data, sample_rate


def save_audio(file_path: Path, data: TensorOrArray, sample_rate: int):
    """
    Saves an audio file

    Parameters
    ----------
    file_path : Path
        Path to the audio file
    data : TensorOrArray
        Audio file to save
    sample_rate : int, optional
        Target sample rate
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


def load_audio_parallel(
    file_paths: List[Path],
    sample_rate: int = None,
    tensor: bool = False,
    device: str = "auto",
    num_workers: int = 4,
) -> List[TensorOrArray]:
    """
    Loads multiple audio files in parallel.

    Parameters
    ----------
    file_path : List[Path]
        List of paths to the audio files
    sample_rate : int, optional
        Target sample rate, by default None
    tensor : bool, optional
        If True loads a torch Tensor, by default False
    device : str, optional
        The device to load the tensor into, by default auto
    num_workers : int, optional
        Number of parallel processes

    Returns
    -------
    List[TensorOrArray]
        list of audios of shape (C, T)
    """
    with Pool(num_workers) as pool:
        _load = lambda x: load_audio(x, sample_rate, False)[0]
        xs = pool.map(_load, file_paths)
    if tensor:
        xs = [Tensor(x).to(device=get_device() if device == "auto" else device) for x in xs]
    return xs


def load_audio_parallel_itr(
    file_paths: List[Path],
    sample_rate: int = None,
    tensor: bool = False,
    device: str = "auto",
    num_workers: int = 4,
) -> Iterator[TensorOrArray]:
    """
    Iterator that loads multiple audio files in parallel.

    Parameters
    ----------
    file_path : List[Path]
        List of paths to the audio files
    sample_rate : int, optional
        Target sample rate, by default None
    tensor : bool, optional
        If True loads a torch Tensor, by default False
    device : str, optional
        The device to load the tensor into, by default auto
    num_workers : int, optional
        Number of parallel processes

    Returns
    -------
    Iterator[TensorOrArray]
        yields audios of shape (C, T)
    """
    n_files = len(file_paths)
    # transforming to generator
    file_paths = (x for x in file_paths)
    for i in range(0, n_files, num_workers):
        files_batch = islice(file_paths, num_workers)
        cache = load_audio_parallel(files_batch, sample_rate, tensor, device, num_workers)
        for x in cache:
            yield x
