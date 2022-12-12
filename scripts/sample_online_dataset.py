from torch import Tensor
from typing import Dict, List
from pathimport import set_module_root
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch


set_module_root("..")
from torch_utils import HDF5OnlineDataset
import torch_utils as tu


def main():
    # argparse
    args = parse_args()

    sr = args.sample_rate

    # seed
    if args.seed != 0:
        np.random.seed(args.seed)

    # paths
    cwd = Path.cwd()
    speech_path = Path(args.speech_dataset)
    noise_path = Path(args.noise_dataset)

    # dataset
    dataset = HDF5OnlineTestDataset(
        dataset_paths=[speech_path, noise_path],
        data_layouts=["x", "x"],
        batch_size=args.count,
        total_items=args.count,
    )

    # sampling
    x = dataset[0]
    x = x[0].type(torch.float32)

    # saving
    _name = lambda i: f"sample_{i}.wav"
    [tu.save_audio(cwd / _name(i), x[i], sr) for i in range(x.shape[0])]


class HDF5OnlineTestDataset(HDF5OnlineDataset):
    def __init__(
        self,
        dataset_paths: List[Path],
        data_layouts: List[List[str]],
        batch_size: int,
        total_items: int,
    ) -> None:
        super().__init__(dataset_paths, data_layouts, batch_size, total_items)

    def transform(self, raw_data: List[Tensor]) -> List[Tensor]:
        s, n = raw_data
        snr = 15
        x = tu.add_noise(s, n, (snr, snr))
        x = tu.scale(x, (0, 0))
        return [x]


def parse_args() -> Dict:
    """
    Parses command line arguments.

    Returns
    -------
    Dict
        Parsed arguments
    """
    desc = "extracts random sequences from an HDF5OnlineDataset for debugging purposes."
    argparser = ArgumentParser(description=desc)
    argparser.add_argument(
        "speech_dataset",
        help=" path to the speech hdf5 dataset",
    )
    argparser.add_argument(
        "noise_dataset",
        help=" path to the noise hdf5 dataset",
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
