__all__ = [
    "load_audio",
    "save_audio",
    "load_audio_parallel",
    "load_audio_parallel_itr",
    "pack_audio_sequences",
]

from typing import Iterator, Sequence, Optional, Tuple
from resampy import resample as resample_np
from torchaudio.functional import resample
from multiprocess import Pool
from itertools import islice
from pathlib import Path
from torch import Tensor
import soundfile as sf
import numpy as np
import torchaudio
import torch


from torch_utilities.utilities import TensorOrArray


def load_audio(
    file_path: Path,
    sample_rate: int = None,
    tensor: bool = False,
    device: Optional[torch.device] = None,
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
    device : Optional[torch.device]
        The device to load the tensor into, by default None

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
        if device is not None:
            data = data.to(device)

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
    file_paths: Sequence[Path],
    sample_rate: int = None,
    tensor: bool = False,
    device: Optional[torch.device] = None,
    num_workers: int = 4,
) -> Sequence[TensorOrArray]:
    """
    Loads multiple audio files in parallel.

    Parameters
    ----------
    file_path : Sequence[Path]
        Sequence of paths to the audio files
    sample_rate : int, optional
        Target sample rate, by default None
    tensor : bool, optional
        If True loads a torch Tensor, by default False
    device : Optional[torch.device]
        The device to load the tensor into, by default None
    num_workers : int, optional
        Number of parallel processes

    Returns
    -------
    Sequence[TensorOrArray]
        Sequence of audios of shape (C, T)
    """
    with Pool(num_workers) as pool:
        _load = lambda x: load_audio(x, sample_rate, False)[0]
        xs = pool.map(_load, file_paths)
    if tensor:
        xs = [Tensor(x, device=device) for x in xs]
    return xs


def load_audio_parallel_itr(
    file_paths: Sequence[Path],
    sample_rate: int = None,
    tensor: bool = False,
    device: Optional[torch.device] = None,
    num_workers: int = 4,
) -> Iterator[TensorOrArray]:
    """
    Iterator that loads multiple audio files in parallel.

    Parameters
    ----------
    file_path : Sequence[Path]
        Sequence of paths to the audio files
    sample_rate : int, optional
        Target sample rate, by default None
    tensor : bool, optional
        If True loads a torch Tensor, by default False
    device : Optional[torch.device]
        The device to load the tensor into, by default None
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
        cache = load_audio_parallel(
            files_batch, sample_rate, tensor, device, num_workers
        )
        for x in cache:
            yield x


def pack_audio_sequences(
    xs: Sequence[Path],
    length: float,
    sample_rate: int,
    channels: int = 1,
    tensor: bool = False,
    delete_last: bool = True,
    num_workers: int = 1,
) -> Iterator[TensorOrArray]:
    """
    Reads from a Sequence of audio filepaths and generate temporal sequences
    of a certain length.

    Parameters
    ----------
    xs : Sequence[Path]
        Sequence of audio filepaths
    length : float
        Length of the sequences is seconds
    sample_rate : int
        Resample at this sample frequency
    channels : int, optional
        Number of channels of the sequences, by default 1
    tensor : bool, optional
        If False returns a numpy ndarray else a torch Tensor, by default False
    delete_last : bool, optional
        If True the last sequence is discarded (since it's typically not complete),
        by default True
    num_workers : int, optional
        Number of parallel processes

    Yields
    ------
    Iterator[TensorOrArray]
        Audio sequence of shape
        (length, C, T)
    """
    # length in samples
    length = int(length * sample_rate)

    # the container of the sequences to be generated
    _zeros = torch.zeros if tensor else np.zeros
    _reset_seq = lambda: _zeros((channels, length))

    # point to the last index consumed
    sample_ptr = 0
    seq_ptr = 0

    # utilities
    _seq_left = lambda: length - seq_ptr
    _sample_left = lambda x: x.shape[1] - sample_ptr
    _copy = lambda x: x.clone() if isinstance(x, Tensor) else x.copy()

    # enables multiprocessing ~ ~ ~ ~ ~
    if num_workers > 1:
        xs = load_audio_parallel_itr(xs, sample_rate, tensor, num_workers=num_workers)
    else:
        xs = (load_audio(x, sample_rate, tensor)[0] for x in xs)
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    seq = _reset_seq()
    for x in xs:
        while _sample_left(x) > 0:
            # copying into the sequence
            delta = min(_seq_left(), _sample_left(x))
            seq[:, seq_ptr : seq_ptr + delta] = x[
                :channels, sample_ptr : sample_ptr + delta
            ]
            seq_ptr += delta
            sample_ptr += delta

            if _seq_left() == 0:
                # sequence complete
                yield _copy(seq)
                seq = _reset_seq()
                seq_ptr = 0

        # sample consumed
        sample_ptr = 0

    if not delete_last:
        yield _copy(seq)
