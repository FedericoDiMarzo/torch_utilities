import random
from typing import Dict
from pathimport import set_module_root
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py

set_module_root("..", prefix=False)
from torch_utils import save_audio


def main():
    # argparse
    args = parse_args()
    sr = args.sample_rate

    # seed
    if args.seed != 0:
        np.random.seed(args.seed)

    # paths
    cwd = Path.cwd()
    dataset_path = Path(args.dataset)

    # reading from the HDF5
    logger.info(f"reading from {dataset_path}")
    with h5py.File(dataset_path, "r") as ds:
        groups = [g for g in ds]
        for _ in range(args.count):
            g = random.choice(groups)
            x = ds[g]["x"]
            sample_idx = np.random.randint(x.shape[0])
            x = x[sample_idx]
            filename = f"{dataset_path.stem}_{g}_sample{sample_idx}.wav"
            save_audio(cwd / filename, x, sr)
            logger.info(f"{filename} extracted")


def parse_args() -> Dict:
    """
    Parses command line arguments.

    Returns
    -------
    Dict
        Parsed arguments
    """
    desc = "Extracts random sequences from an HDF5 for debugging purposes."
    argparser = ArgumentParser(description=desc)
    argparser.add_argument(
        "dataset",
        help=" path to the hdf5 dataset",
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
        help="Loads first channel only",
    )
    argparser.add_argument(
        "--count",
        default=1,
        type=int,
        help="number of samples to extract",
    )
    argparser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed",
    )
    return argparser.parse_args()


if __name__ == "__main__":
    main()
