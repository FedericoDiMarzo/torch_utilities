from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler
from typing import List, Optional, Tuple
from pathimport import set_module_root
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
import numpy as np
import itertools
import torch
import h5py

set_module_root(".")
from torch_utilities.common import to_numpy, split_complex

# export list
__all__ = [
    "WeakShufflingSampler",
    "HDF5Dataset",
    "get_hdf5_dataloader",
    "HDF5OnlineDataset",
    "get_dataset_statistics",
    "plot_dataset_statistics",
]


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
        data_layout: List[str],
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
        data_layout : List[str]
            List describing the layout of the data
            inside each group. For an input-label1-label2
            dataset the list would be ["input", "label1", "label2"]
        """
        super().__init__()
        self.dataset_path = dataset_path.absolute()
        self.data_layout = data_layout
        self.dataset_file = h5py.File(self.dataset_path, "r")
        self.groups = list(self.dataset_file.keys())
        self.group_batch_len = self.dataset_file[self.groups[0]][self.data_layout[0]].shape[0]
        self.data_layout = data_layout
        self._cache = None
        self._cache_idx = None

    def __len__(self):
        return len(self.groups) * self.group_batch_len

    def __getitem__(self, idx) -> List[Tensor]:
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
        a, b = idx[0] % gbl, (idx[-1] % gbl) + 1
        data = [x[a:b] for x in self._cache]

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
        cast = lambda x: torch.from_numpy(np.array(x)).to("cpu")
        data = [cast(g[k]) for k in self.data_layout]
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
    dataset_path: Path,
    data_layout: List[str],
    batch_size: int = None,
    **dataloader_kwargs: dict,
) -> DataLoader:
    """
    Create a dataloader binded to a HDF5Dataset.

    Parameters
    ----------
    dataset : Path
        Path to the HDF5 Dataset
    data_layout : List[str]
            List describing the layout of the data
            inside each group. For an input-label1-label2
            dataset the list would be ["input", "label1", "label2"]
    batch_size : int, optional
        Batch size of the dataloader, by default HDF5Dataset.group_batch_len
    dataloader_kwargs : dict, optional
        DataLoader arguments, by default {}
    """
    if batch_size is None:
        batch_size = dataset.group_batch_len

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataset = HDF5Dataset(dataset_path, data_layout)

    sampler = BatchSampler(
        sampler=WeakShufflingSampler(
            dataset=dataset,
            batch_size=batch_size,
        ),
        batch_size=batch_size,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        **dataloader_kwargs,
    )

    return dataloader


class HDF5OnlineDataset(Dataset):
    def __init__(
        self,
        dataset_paths: List[Path],
        data_layouts: List[List[str]],
        batch_size: int = 16,
        total_items: int = 1024,
    ) -> None:
        """
        Dataset based on online generation.
        Data can be combined after retrieving it
        from source HDF5Datasets.

        Parameters
        ----------
        dataset_paths : List[Path]
            List of paths to the HDF5Datasets h5
        data_layouts : List[List[str]]
            Data layouts for all the HDF5Datasets
        batch_size : int, optional
            Batch size of each raw data, by default 16
        total_items : int, optional
            Total item in an epoch, by default 1024
        """
        super(HDF5OnlineDataset).__init__()
        self.datasets_paths = dataset_paths
        self.data_layouts = data_layouts
        self.data_layouts = data_layouts
        self.batch_size = batch_size
        self.total_items = total_items

        # error handling
        err_msg = f"total_items ({total_items}) must be divisible by batch_size ({batch_size})"
        assert (self.total_items % self.batch_size) == 0, err_msg

        # static datasets
        self.source_datasets = self._get_datasets()

    def __len__(self) -> int:
        return self.total_items // self.batch_size

    def transform(self, raw_data: List[Tensor]) -> List[Tensor]:
        """
        Override this method to process the raw data.

        Parameters
        ----------
        raw_data : List[Tensor]
            Raw data from the HDF5Datasets

        Returns
        -------
        List[Tensor]
            Processed data
        """
        return raw_data

    def _get_rand_batch(self, dataset: HDF5Dataset) -> List[Tensor]:
        """
        Get a random batch from an HDF5Dataset

        Parameters
        ----------
        dataset : HDF5Dataset
            Target HDF5Dataset

        Returns
        -------
        List[Tensor]
            Batches from the HDF5Datasets
        """
        # getting start and end points
        ds_len = len(dataset)
        total_batches = ds_len // self.batch_size
        r_int = np.random.randint(0, total_batches)
        start = r_int * self.batch_size
        end = (r_int + 1) * self.batch_size
        indices = list(range(start, end))

        # getting the random selection
        ds_outs = dataset[indices]
        return ds_outs

    def __getitem__(self, _) -> List[Tensor]:
        outs = [self._get_rand_batch(ds) for ds in self.source_datasets]
        outs = itertools.chain.from_iterable(outs)
        outs = list(outs)
        outs = self.transform(outs)
        return outs

    def _get_datasets(self) -> List[HDF5Dataset]:
        """
        Gets the static datasets.

        Returns
        -------
        List[HDF5Dataset]
            List of static datasets
        """
        ds = [HDF5Dataset(p, l) for p, l in zip(self.datasets_paths, self.data_layouts)]
        return ds


def get_dataset_statistics(
    dataloader: DataLoader,
    iterations: int,
    n_hist_bins: int = 100,
    verbose: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, float, float, float]]:
    """
    Calculate the statistics for the data obtained
    from a dataset.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to iterate
    iterations : int
        Number of iterations
    n_hist_bins : int, optional
        Number of histogram bins
    verbose : bool, optional
        If True shows a progress bar for the iterations,
        by default False

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, float, float, float, float]]
        List of statistics for each dataset output
        each tuple contains
        (hist_vals, hist_bins, min, max, mean, var)
    """
    ds = iter(dataloader)
    _range = tqdm(range(iterations)) if verbose else range(iterations)
    outs = [ds.__next__() for i in _range]
    n_items = len(outs[0])

    stats = []
    for i in range(n_items):
        x = torch.stack([o[i] for o in outs])
        x = x.flatten()
        x = x.float()
        x = split_complex(x) if torch.is_complex(x) else x
        hist_vals, hist_bins = [to_numpy(y) for y in x.histogram(n_hist_bins, density=True)]
        hist_vals /= n_hist_bins
        _min = x.min().item()
        _max = x.max().item()
        _mean = x.mean().item()
        _var = x.var().item()
        stats.append((hist_vals, hist_bins, _min, _max, _mean, _var))

    return stats


def plot_dataset_statistics(
    dataloader: DataLoader,
    iterations: int,
    n_hist_bins: int = 100,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    verbose: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, float, float, float, float]]:
    """
    Plots the statistics for the data obtained
    from a dataset.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to iterate
    iterations : int
        Number of iterations
    n_hist_bins : int, optional
        Number of histogram bins
    labels : Optional[List[str]]
        Names of the dataset outputs
    figsize : Tuple[int, int], optional
        Size of the plot, by default (8, 6)
    verbose : bool, optional
        If True shows a progress bar for the iterations,
        by default False

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, float, float, float]]
        List of statistics for each dataset output
        each tuple contains
        (hist_vals, hist_bins, min, max, mean, var)
    """
    import matplotlib.pyplot as plt

    all_stats = get_dataset_statistics(
        dataloader,
        iterations,
        n_hist_bins,
        verbose,
    )

    if labels == None:
        # default labels
        labels = [f"feature_{i}" for i in range(len(all_stats))]

    for stats, name in zip(all_stats, labels):
        hist_vals, hist_bins, min, max, mean, var = stats
        bar_width = np.abs(hist_bins[1] - hist_bins[0])
        plt.figure(figsize=figsize)
        plt.title(f"{name} distribution")
        plt.bar(hist_bins[:-1], hist_vals, width=bar_width)
        plt.vlines(x=mean, ymin=0, ymax=1, colors="red", label="mean")
        plt.ylim([0, hist_vals.max()])
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

    return all_stats
