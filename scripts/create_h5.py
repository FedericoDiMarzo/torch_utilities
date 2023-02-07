from typing import Dict, List
from pathimport import set_module_root
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np
import itertools
import h5py
import sys

set_module_root("..", prefix=False)
from torch_utilities import load_audio, random_trim, fade_sides, trim_silence, pack_audio_sequences


def main():
    # argparse
    args = parse_args()
    sr = args.sample_rate
    gbl = args.group_batch_len
    thr = args.trim_silence_threshold
    mono = args.mono
    fade_direction = "right" if args.right_fade_only else "both"
    no_rand_trim = args.no_rand_trim
    length = args.len

    # calculating dataset path
    cwd = Path.cwd()
    if Path(args.dataset_name).is_absolute():
        dataset_path = Path(args.dataset_name)
    else:
        dataset_path = cwd / args.dataset_name

    # filepaths = list(sys.stdin)
    filepaths = list(sys.stdin)

    # writing into the HDF5
    if args.pack_samples:
        pack_samples_in_h5(
            dataset_path,
            filepaths,
            sr,
            gbl,
            mono,
            length,
        )
    else:
        isolated_samples_in_h5(
            dataset_path,
            filepaths,
            sr,
            gbl,
            mono,
            thr,
            length,
            fade_direction,
            no_rand_trim,
        )


def pack_samples_in_h5(
    dataset_path: Path,
    filepaths: List[str],
    sample_rate: int,
    group_batch_len: int,
    mono: bool,
    length: int,
) -> None:
    """
    Fill an HDF5 with multiple samples concatenated
    for each batch.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset
    filepaths : List[str]
        List of audio files
    sample_rate : int
        Sample frequency in Hz
    group_batch_len : int
        batch dimension of each group
    mono : bool
        If True force to mono taking the first channel
    threshold : float
        Silence trimming threshold
    length : float
        Sequence length
    """
    # getting the number of channels
    filepaths = [Path(x) for x in filepaths]
    channels = 1 if mono else load_audio(filepaths[0])[0].shape[0]

    with h5py.File(dataset_path, "w") as ds:
        filepaths = tqdm(filepaths)
        itr = pack_audio_sequences(filepaths, length, sample_rate, channels)
        for i in itertools.count():
            try:
                xs = [x for x in itertools.islice(itr, group_batch_len)]
                if len(xs) < group_batch_len:
                    break
                x = np.stack(xs)
                g = ds.create_group(f"group_{i}")
                g.create_dataset("x", data=x)
            except StopIteration:
                # no more sequences
                break


def isolated_samples_in_h5(
    dataset_path: Path,
    filepaths: List[str],
    sample_rate: int,
    group_batch_len: int,
    mono: bool,
    thr: float,
    length: int,
    fade_direction: str,
    no_rand_trim: bool,
) -> None:
    """
    Fill an HDF5 with one sample per batch.
    Applies a fade out/in depending on the paramenters.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset
    filepaths : List[str]
        List of audio files
    sample_rate : int
        Sample frequency in Hz
    group_batch_len : int
        batch dimension of each group
    mono : bool
        If True force to mono taking the first channel
    threshold : float
        Silence trimming threshold
    length : float
        Sequence length
    fade_direction : str
        One between "both" and "right"
    no_rand_trim : bool
        If True the samples are always taken from their beginning,
        by default False
    """

    # checking that group_batch_len divides the inputs files
    input_reminder = len(filepaths) % group_batch_len
    if input_reminder != 0:
        warn_msg = f"{input_reminder} samples will be removed (group_batch_len=={group_batch_len}, total_samples=={len(filepaths)})"
        logger.warning(warn_msg)

    groups = len(filepaths) // group_batch_len
    _transform = lambda x: transform(
        x, sample_rate, mono, thr, length, fade_direction, no_rand_trim
    )
    with h5py.File(dataset_path, "w") as ds:
        for i in tqdm(range(groups)):
            selection = filepaths[i * group_batch_len : (i + 1) * group_batch_len]
            selection = [path.rstrip("\n") for path in selection]
            tracks = [load_audio(path, sample_rate)[0] for path in selection]
            tracks_trimmed = [_transform(x) for x in tracks]
            if mono:
                tracks = [x[:0] for x in tracks]
            x = np.stack(tracks_trimmed)
            g = ds.create_group(f"group_{i}")
            g.create_dataset("x", data=x)


def transform(
    x: np.ndarray,
    sample_rate: int,
    mono: bool,
    threshold: float,
    length: float,
    fade_direction: str,
    no_rand_trim: bool,
) -> np.ndarray:
    """
    Trims the silence from the sides of the sample, cut a section to fit
    the length of the sequence and applies a fade.

    Parameters
    ----------
    x : np.ndarray
        Input signal of shape (C, T)
    sample_rate : int
        Sample frequency in Hz
    mono : bool
        If True force to mono taking the first channel
    threshold : float
        Silence trimming threshold
    length : float
        Sequence length
    fade_direction : str
        One between "both" and "right"
    no_rand_trim : bool
        If True the samples are always taken from their beginning,
        by default False

    Returns
    -------
    np.ndarray
        Trimmed sample
    """
    if mono:
        x = x[0][None, ...]
    x = trim_silence(x, threshold)
    if no_rand_trim:
        # trimming the sample from the beginning
        C, T = x.shape
        len_samples = int(length * sample_rate)
        tmp = np.zeros((C, len_samples))
        end = min(T, len_samples)
        tmp[..., :T] = x[..., :end]
        x = tmp
    else:
        x = random_trim(x, sample_rate, length)
    x = fade_sides(x, direction=fade_direction)
    return x


def parse_args() -> Dict:
    """
    Parses command line arguments.

    Returns
    -------
    Dict
        Parsed arguments
    """
    desc = "Converts a list of absolute wav paths passed through stdin to an HDF5 dataset."
    argparser = ArgumentParser(description=desc)
    argparser.add_argument(
        "dataset_name",
        help="name or path to the new hdf5 dataset",
    )
    argparser.add_argument(
        "--group_batch_len",
        default=32,
        type=int,
        help="batch dimension of each group",
    )
    argparser.add_argument(
        "--sample_rate",
        default=16000,
        type=int,
        help="resample to the sample rate indicated",
    )
    argparser.add_argument(
        "--multichannel",
        dest="mono",
        default=True,
        action="store_false",
        help="Loads the samples with their original channels count",
    )
    argparser.add_argument(
        "--len",
        default=3,
        type=float,
        help="duration in seconds inside the datasets, by default 3 s",
    )
    argparser.add_argument(
        "--trim_silence_threshold",
        default=-35,
        type=float,
        help="threshold for the silence trimming, by default -35 dB",
    )
    argparser.add_argument(
        "--right_fade_only",
        default=False,
        action="store_true",
        help="if set, a fade is performed on the right side of the samples only",
    )
    argparser.add_argument(
        "--no_rand_trim",
        default=False,
        action="store_true",
        help="if set, the samples are always taken from their beginning",
    )
    argparser.add_argument(
        "--pack_samples",
        default=False,
        action="store_true",
        help="if set, the samples are packed to fill the sequences without any fade",
    )

    return argparser.parse_args()


if __name__ == "__main__":
    main()
