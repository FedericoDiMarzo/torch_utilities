from itertools import product
from torch import Tensor
import numpy as np
import unittest
import torch

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

    def delete_tmp(self):
        self.tmp_file.unlink(missing_ok=True)

    def test_load_audio(self):
        mono = (False, True)
        sample_rate = (None, 16000, 48000)
        tensor = (False, True)
        device = ("cpu", "auto")
        params = product(
            mono,
            sample_rate,
            tensor,
            device,
        )
        for p in params:
            (m, sr, t, d) = p
            default_sr = 16000
            with self.subTest(p=p):
                file = self.mono_file if m else self.stereo_file
                x, x_sr = tu.load_audio(file, sr, t, d)
                self.assertEqual(len(x.shape), 2)
                self.assertEqual(x.shape[0], 1 if m else 2)
                self.assertEqual(type(x), Tensor if t else np.ndarray)
                self.assertEqual(x_sr, sr if sr is not None else default_sr)
                if t:
                    expected_device = "cpu" if d == "cpu" else tu.get_device()
                    self.assertEqual(x.device.type, expected_device)

    def test_save_audio(self):
        mono = (False, True)
        sample_rate = (16000, 48000)
        tensor = (False, True)
        params = product(
            mono,
            sample_rate,
            tensor,
        )
        for p in params:
            (m, sr, t) = p
            with self.subTest(p=p):
                ch = 1 if m else 2
                x = torch.zeros((ch, sr)) if t else np.zeros((ch, sr))
                tu.save_audio(self.tmp_file, x, sr)
                self.delete_tmp()

    def test_load_audio_parallel(self):
        mono = (False, True)
        sample_rate = (16000, 48000)
        tensor = (False, True)
        device = ("cpu", "auto")
        num_workers = (1, 4)
        params = product(
            mono,
            sample_rate,
            tensor,
            device,
            num_workers,
        )
        for p in params:
            (m, sr, t, d, w) = p
            n_files = 4
            with self.subTest(p=p):
                name = "mono.wav" if m else "stereo.wav"
                filepaths = [get_test_data_dir() / name for _ in range(n_files)]
                xs = tu.load_audio_parallel(filepaths, sr, t, d, w)
                self.assertEqual(len(xs), n_files)
                for x in xs:
                    self.assertEqual(type(x), Tensor if t else np.ndarray)
                    self.assertEqual(x.shape[0], 1 if m else 2)

    def test_load_audio_parallel_itr(self):
        mono = (False, True)
        sample_rate = (16000, 48000)
        tensor = (False, True)
        device = ("cpu", "auto")
        num_workers = (1, 2)
        params = product(
            mono,
            sample_rate,
            tensor,
            device,
            num_workers,
        )
        for p in params:
            (m, sr, t, d, w) = p
            n_files = 4
            with self.subTest(p=p):
                name = "mono.wav" if m else "stereo.wav"
                filepaths = [get_test_data_dir() / name for _ in range(n_files)]
                xs = tu.load_audio_parallel_itr(filepaths, sr, t, d, w)
                for i, x in enumerate(xs):
                    self.assertEqual(type(x), Tensor if t else np.ndarray)
                    self.assertEqual(x.shape[0], 1 if m else 2)
                self.assertEqual(i + 1, n_files)


if __name__ == "__main__":
    unittest.main(verbosity=2)
