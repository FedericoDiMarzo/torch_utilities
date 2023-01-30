from itertools import product
from pathlib import Path
from pathimport import set_module_root
from torch import Tensor
from typing import List
import numpy as np
import unittest
import torch

set_module_root("../torch_utils")
from tests.generate_test_data import get_test_data_dir
import torch_utilities as tu


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("cpu")
    torch.set_grad_enabled(False)


class TestHDF5DataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.data_layout = ["x", "y_true"]
        self.hdf5_path = get_test_data_dir() / "dataset.hdf5"
        self.dataset = tu.HDF5Dataset(self.hdf5_path, self.data_layout)

    def dummy_input(self, g_idx):
        return torch.ones((16, 8)) * g_idx

    def test_weak_shuffling_len(self):
        batch_size = 16
        sampler = tu.WeakShufflingSampler(self.dataset, batch_size)
        self.assertEqual(len(sampler), batch_size)

    def test_hdf5dataset_cache(self):
        group_len = 16

        self.assertEqual(self.dataset._cache, None)
        self.assertEqual(self.dataset._cache_idx, None)

        self.dataset[[0, 1]]
        self.assertTrue(torch.all(self.dataset._cache[0] == self.dummy_input(0)))
        self.assertTrue(torch.all(self.dataset._cache[1] == self.dummy_input(0)))
        self.assertEqual(self.dataset._cache_idx, 0)

        self.dataset[[group_len * 3, group_len * 3 + 1]]
        self.assertTrue(torch.all(self.dataset._cache[0] == self.dummy_input(3)))
        self.assertTrue(torch.all(self.dataset._cache[1] == self.dummy_input(3)))
        self.assertEqual(self.dataset._cache_idx, group_len * 3)

    def test_hdf5dataset_raise_idx_not_list(self):
        with self.assertRaises(RuntimeError):
            self.dataset[0]

    def test_hdf5dataset_raise_idx_len_too_long(self):
        with self.assertRaises(RuntimeError):
            self.dataset[list(range(17))]

    def test_hdf5dataset_raise_idx_len_not_divisor(self):
        with self.assertRaises(RuntimeError):
            self.dataset[[1, 2, 3]]

    def test_hdf5_dataloader_full_batch(self):
        dataloader = tu.get_hdf5_dataloader(self.hdf5_path, self.data_layout, 16)
        data = [x for x in dataloader]
        self.assertEqual(len(data), 10)

    def test_hdf5_dataloader_half_batch(self):
        dataloader = tu.get_hdf5_dataloader(self.hdf5_path, self.data_layout, 8)
        data = [x for x in dataloader]
        self.assertEqual(len(data), 20)

    def test_get_dataset_statistics(self):
        hist_bins = 10
        dataloader = tu.get_hdf5_dataloader(self.hdf5_path, self.data_layout, 16)
        iterations = len(dataloader)
        stats = tu.get_dataset_statistics(dataloader, iterations, hist_bins)
        hist_vals, hist_bins, min, max, mean, var = stats[0]
        self.assertEqual(len(stats), 2)
        self.assertAlmostEqual(np.sum(hist_vals), 1)


class HDF5OnlineDatasetTesting(tu.HDF5OnlineDataset):
    def __init__(
        self,
        dataset_paths: List[Path],
        data_layouts: List[List[str]],
        batch_size: int,
        total_items: int,
    ) -> None:
        super().__init__(dataset_paths, data_layouts, batch_size, total_items)

    def transform(self, raw_data: List[Tensor]) -> List[Tensor]:
        return raw_data


class TestHDF5OnlineDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.data_layout = ["x", "y_true"]
        self.hdf5_path = get_test_data_dir() / "dataset.hdf5"

    def test_assert(self):
        batch_len = (16, 32)
        total_items = (77, 171)
        for b, t in product(batch_len, total_items):
            with self.subTest(b=b, t=t), self.assertRaises(AssertionError):
                HDF5OnlineDatasetTesting(
                    dataset_paths=[self.hdf5_path],
                    data_layouts=[self.data_layout],
                    batch_size=b,
                    total_items=t,
                )

    def test_len(self):
        batch_len = (8, 16)
        total_items = (16, 32)
        for b, t in product(batch_len, total_items):
            with self.subTest(b=b, t=t):
                ds = HDF5OnlineDatasetTesting(
                    dataset_paths=[self.hdf5_path],
                    data_layouts=[self.data_layout],
                    batch_size=b,
                    total_items=t,
                )
                ds_len = len(ds)
                self.assertEqual(ds_len, t // b)

    def test_get_datasets(self):
        ds = HDF5OnlineDatasetTesting(
            dataset_paths=[self.hdf5_path, self.hdf5_path],
            data_layouts=[self.data_layout, self.data_layout],
            batch_size=8,
            total_items=1024,
        )

        datasets = ds._get_datasets()
        for d in datasets:
            self.assertEqual(type(d), tu.HDF5Dataset)

    def test_get_rand_batch(self):
        batch_size = 8
        ds = HDF5OnlineDatasetTesting(
            dataset_paths=[self.hdf5_path],
            data_layouts=[self.data_layout],
            batch_size=batch_size,
            total_items=64,
        )

        datasets = ds._get_datasets()
        for d in datasets:
            raw_data = ds._get_rand_batch(d)
            for batch in raw_data:
                self.assertEqual(batch.shape[0], batch_size)

    def test_get_dataset_statistics(self):
        hist_bins = 10
        dataloader = tu.get_hdf5_dataloader(self.hdf5_path, self.data_layout, 16)
        iterations = len(dataloader)
        stats = tu.get_dataset_statistics(dataloader, iterations, hist_bins)
        hist_vals, hist_bins, min, max, mean, var = stats[0]
        self.assertEqual(len(stats), 2)
        self.assertAlmostEqual(np.sum(hist_vals), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
