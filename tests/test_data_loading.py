import unittest
from pathimport import set_module_root
import numpy as np
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as tu
from tests.generate_test_data import get_test_data_dir
from torch_utils import set_device, get_device

torch.manual_seed(984)
np.random.seed(901)
set_device("auto")


class TestHDF5DataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

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


if __name__ == "__main__":
    unittest.main()
