from pathimport import set_module_root
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py
import sys

set_module_root("..")
from torch_utils.common import load_audio, trim, fade_sides


def main(args):
    gbl = args.group_batch_len
    sr = args.sample_rate

    # calculating dataset path
    cwd = Path.cwd()
    if Path(args.dataset_name).is_absolute():
        dataset_path = Path(args.dataset_name)
    else:
        dataset_path = cwd / args.dataset_name

    # checking that group_batch_len divides the inputs files
    stdin = list(sys.stdin)
    input_reminder = len(stdin) % gbl
    if input_reminder != 0:
        warn_msg = f"{input_reminder} will be removed (group_batch_len=={gbl})"
        logger.warning(warn_msg)

    # writing into the HDF5
    groups = len(stdin) // gbl
    transform = lambda x: fade_sides(trim(x, sr, args.len))
    with h5py.File(dataset_path, "w") as ds:
        for i in tqdm(range(groups)):
            selection = stdin[i * gbl : (i + 1) * gbl]
            selection = [path.rstrip("\n") for path in selection]
            tracks = [load_audio(path, sr)[0] for path in selection]
            tracks_trimmed = [transform(x) for x in tracks]
            if args.mono:
                tracks = [x[:0] for x in tracks]
            x = np.stack(tracks_trimmed)
            g = ds.create_group(f"group_{i}")
            g.create_dataset("x", data=x)


if __name__ == "__main__":
    desc = "converts a list of absolute wav paths passed through stdin to an HDF5 dataset."
    argparser = ArgumentParser(description=desc)
    argparser.add_argument(
        "dataset_name",
        help="name or path to the new hdf5 dataset",
    )
    argparser.add_argument(
        "--group_batch_len",
        default=32,
        help="batch dimension of each group",
    )
    argparser.add_argument(
        "--sample_rate",
        default=16000,
        help="resample to the sample rate indicated",
    )
    argparser.add_argument(
        "--mono",
        default=True,
        action="store_false",
        help="True loads first channel only",
    )
    argparser.add_argument(
        "--len",
        default=3,
        help="duration in seconds inside the datasets",
    )
    args = argparser.parse_args()
    main(args)
