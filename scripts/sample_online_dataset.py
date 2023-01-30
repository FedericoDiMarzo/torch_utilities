from pathimport import set_module_root
from argparse import ArgumentParser
from typing import Dict, List
from loguru import logger
from torch import Tensor
from pathlib import Path
import numpy as np
import torch


set_module_root("..", prefix=False)
from torch_utilities import HDF5OnlineDataset
import torch_utilities.augmentation as aug
import torch_utilities as tu


def main():
    # argparse
    args = parse_args()

    sr = args.sample_rate
    snr = args.snr

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
        batch_size=args.count,
        total_items=args.count,
        snr=snr,
    )

    # sampling
    x = dataset[0]
    x = x[0].type(torch.float32)

    # saving
    _name = lambda i: f"sample_{i}.wav"
    [tu.save_audio(cwd / _name(i), x[i], sr) for i in range(x.shape[0])]
    logger.info(f"{args.count} samples saved")


class HDF5OnlineTestDataset(HDF5OnlineDataset):
    def __init__(
        self,
        dataset_paths: List[Path],
        batch_size: int,
        total_items: int,
        snr: float,
    ) -> None:
        super().__init__(dataset_paths, [["x"], ["x"]], batch_size, total_items)
        self.snr = snr

    def transform(self, raw_data: List[Tensor]) -> List[Tensor]:
        s, n = raw_data
        x = aug.add_noise(s, n, (self.snr, self.snr))
        x = aug.scale(x, (0, 0))
        return [x]


def parse_args() -> Dict:
    """
    Parses command line arguments.

    Returns
    -------
    Dict
        Parsed arguments
    """
    desc = (
        "extracts random sequences from a HDF5OnlineTestDataset for debugging purposes."
    )
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
        help="loads first channel only",
    )
    argparser.add_argument(
        "--snr",
        type=float,
        default=15,
        help="Signal to Noise Ratio",
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
