import unittest
from pathimport import set_module_root
from torch import Tensor
import numpy as np
import torch

set_module_root("../torch_utils")
from tests.generate_test_data import get_test_data_dir
import torch_utilities as tu


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.mono_file = get_test_data_dir() / "mono.wav"
        self.stereo_file = get_test_data_dir() / "stereo.wav"
        self.tmp_file = get_test_data_dir() / "tmp.wav"
        self.tmp_sr = 16000

    def delete_tmp(self):
        self.tmp_file.unlink(missing_ok=True)

    def test_mono_read(self):
        x = tu.load_audio(self.mono_file)[0]
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(np.ndarray, type(x))

    def test_stereo_read(self):
        x = tu.load_audio(self.stereo_file)[0]
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(np.ndarray, type(x))

    def test_tensor_mono_read(self):
        x = tu.load_audio(self.mono_file, tensor=True)[0]
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(Tensor, type(x))

    def test_tensor_stereo_read(self):
        x = tu.load_audio(self.stereo_file, tensor=True)[0]
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(Tensor, type(x))

    def test_resample_read(self):
        x = tu.load_audio(self.mono_file, 48000)

    def test_mono_save(self):
        x = np.zeros((1, self.tmp_sr))
        tu.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_stereo_save(self):
        x = np.zeros((2, self.tmp_sr))
        tu.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_tensor_mono_save(self):
        x = torch.zeros((1, self.tmp_sr))
        tu.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_tensor_stereo_save(self):
        x = torch.zeros((2, self.tmp_sr))
        tu.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_tensor_resample_read(self):
        x = tu.load_audio(self.mono_file, 48000, tensor=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
