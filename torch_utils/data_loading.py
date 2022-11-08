from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler
from pathimport import set_module_root
from typing import List
from pathlib import Path
from torch import Tensor
import numpy as np
import torch
import h5py

set_module_root(".", prefix=True)
from torch_utils.common import set_device

# export list
__all__ = [
    "WeakShufflingSampler",
    "HDF5Dataset",
    "get_hdf5_dataloader",
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
            List describing the layout of the data
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

    def __getitem__(self, idx) -> List[Tensor]:
        set_device("cpu")
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
        set_device("auto")

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
        cast = lambda x: torch.from_numpy(np.array(x))
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
        batch_size = dataset_path.group_batch_len

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataset = HDF5Dataset(dataset_path, data_layout)

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
