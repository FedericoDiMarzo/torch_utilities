from typing import Callable
import unittest
from pathlib import Path
from pathimport import set_module_root
from torch import Tensor
import numpy as np
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as tu
from tests.generate_test_data import get_test_data_dir
from torch_utils.common import repeat_test, set_auto_device

torch.manual_seed(984)
np.random.seed(901)
set_auto_device()


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.config_path = get_test_data_dir() / "test.yml"
        self.config = tu.Config(self.config_path)

    def test_get_str(self):
        y_true = "test"
        y_test = self.config.get("section1", "param1", str)
        self.assertEqual(y_true, y_test)

    def test_get_int(self):
        y_true = 42
        y_test = self.config.get("section1", "param2", int)
        self.assertEqual(y_true, y_test)

    def test_get_list(self):
        y_true = ["a", "b", "c"]
        y_test = self.config.get("section1", "param3", list)
        self.assertEqual(y_true, y_test)

    def test_get_float(self):
        y_true = 12.43
        y_test = self.config.get("section2", "param4", float)
        self.assertAlmostEqual(y_true, y_test)

    def test_default_none(self):
        y_true = None
        y_test = self.config.get("section1", "param10")
        self.assertEqual(y_true, y_test)

    def test_default(self):
        y_true = "default"
        y_test = self.config.get("section1", "param10", default="default", _type=int)
        self.assertEqual(y_true, y_test)


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

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


class TestSTFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.sample_rate = 16000
        self.framesize_ms = 10
        self.x_np = np.zeros((self.sample_rate))
        self.x_pt = torch.zeros((self.sample_rate))
        self.x_np_stft = np.zeros((50, 321)).astype(complex)
        self.x_pt_stft = torch.zeros((50, 321)).type(torch.cfloat)

    def test_mono_stft(self):
        x = self.x_np
        for ovs, bins in zip([1, 4], [81, 321]):
            x_stft = tu.stft(
                x,
                sample_rate=self.sample_rate,
                framesize_ms=self.framesize_ms,
                frame_oversampling=ovs,
            )
            self.assertEqual(len(x_stft.shape), 2)
            self.assertEqual(x_stft.shape[1], bins)

    def test_multich_stft(self):
        x = np.stack([self.x_np] * 4)
        for ovs, bins in zip([1, 4], [81, 321]):
            x_stft = tu.stft(
                x,
                sample_rate=self.sample_rate,
                framesize_ms=self.framesize_ms,
                frame_oversampling=ovs,
            )
            self.assertEqual(len(x_stft.shape), 3)
            self.assertEqual(x_stft.shape[0], 4)
            self.assertEqual(x_stft.shape[2], bins)

    def test_mono_istft(self):
        x = self.x_np_stft
        x_istft = tu.istft(
            x,
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
            frame_oversampling=4,
        )
        self.assertEqual(len(x_istft.shape), 1)

    def test_multich_istft(self):
        x = np.stack([self.x_np_stft] * 4)
        x_istft = tu.istft(
            x,
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
            frame_oversampling=4,
        )
        self.assertEqual(len(x_istft.shape), 2)
        self.assertEqual(x_istft.shape[0], 4)

    def test_inversion(self):
        eps = 1e-6
        x_len = 16000
        x = np.random.uniform(-1, 1, x_len)
        kwargs = dict(
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
        )
        x_stft = tu.stft(x, **kwargs)
        x_hat = tu.istft(x_stft, **kwargs)

        # skipping the fist frame (due to the windowing artifacts)
        x_len -= 1
        x = x[1:]
        x_hat = x_hat[1:]
        err = np.mean(np.abs(x - x_hat[:x_len]))

        self.assertLess(err, eps)

    def test_tensor_mono_stft(self):
        x = self.x_pt
        for ovs, bins in zip([1, 4], [81, 321]):
            x_stft = tu.stft(
                x,
                sample_rate=self.sample_rate,
                framesize_ms=self.framesize_ms,
                frame_oversampling=ovs,
            )
            self.assertEqual(len(x_stft.shape), 2)
            self.assertEqual(x_stft.shape[1], bins)

    def test_tensor_multich_stft(self):
        x = torch.stack([self.x_pt] * 4)
        for ovs, bins in zip([1, 4], [81, 321]):
            x_stft = tu.stft(
                x,
                sample_rate=self.sample_rate,
                framesize_ms=self.framesize_ms,
                frame_oversampling=ovs,
            )
            self.assertEqual(len(x_stft.shape), 3)
            self.assertEqual(x_stft.shape[0], 4)
            self.assertEqual(x_stft.shape[2], bins)

    def test_tensor_mono_istft(self):
        x = self.x_pt_stft
        x_istft = tu.istft(
            x,
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
            frame_oversampling=4,
        )
        self.assertEqual(len(x_istft.shape), 1)

    def test_tensor_multich_istft(self):
        x = torch.stack([self.x_pt_stft] * 4)
        x_istft = tu.istft(
            x,
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
            frame_oversampling=4,
        )
        self.assertEqual(len(x_istft.shape), 2)
        self.assertEqual(x_istft.shape[0], 4)


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
        dataloader = tu.get_hdf5_dataloader(self.dataset, 16)
        data = [x for x in dataloader]
        self.assertEqual(len(data), 10)

    def test_hdf5_dataloader_half_batch(self):
        dataloader = tu.get_hdf5_dataloader(self.dataset, 8)
        data = [x for x in dataloader]
        self.assertEqual(len(data), 20)


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_db(self):
        x = np.random.uniform(0, 1, 100)
        y = tu.db(x)
        x_hat = tu.invert_db(y)
        self.assertTrue(np.allclose(x, x_hat))

    def test_db_tensor(self):
        x = torch.rand(100)
        y = tu.db(x)
        x_hat = tu.invert_db(y)
        self.assertTrue(torch.allclose(x, x_hat))

    def test_power(self):
        x = np.ones(100)
        x[50:] = -1
        self.assertAlmostEqual(tu.power(x), 100)

    def test_power_multichannel(self):
        x = np.ones((2, 100))
        x[:, 50:] = -1
        y = tu.power(x)
        self.assertAlmostEqual(y[0], 100)
        self.assertAlmostEqual(y[1], 100)

    def test_power_tensor(self):
        x = torch.ones(100)
        x[50:] = -1
        self.assertAlmostEqual(tu.power(x), 100)

    def test_power_multichannel_tensor(self):
        x = torch.ones((2, 100))
        x[:, 50:] = -1
        y = tu.power(x)
        self.assertAlmostEqual(y[0], 100)
        self.assertAlmostEqual(y[1], 100)

    def test_energy(self):
        x = np.ones(100)
        x[50:] = -1
        self.assertAlmostEqual(tu.energy(x), 1)

    def test_energy_multichannel(self):
        x = np.ones((2, 100))
        x[:, 50:] = -1
        y = tu.energy(x)
        self.assertAlmostEqual(y[0], 1)
        self.assertAlmostEqual(y[1], 1)

    def test_energy_tensor(self):
        x = torch.ones(100)
        x[50:] = -1
        self.assertAlmostEqual(tu.energy(x), 1)

    def test_energy_multichannel_tensor(self):
        x = torch.ones((2, 100))
        x[:, 50:] = -1
        y = tu.energy(x)
        self.assertAlmostEqual(y[0], 1)
        self.assertAlmostEqual(y[1], 1)

    def test_rms(self):
        x = np.ones(100)
        x[50:] = -1
        self.assertAlmostEqual(tu.rms(x), 1)

    def test_rms_multichannel(self):
        x = np.ones((2, 100))
        x[:, 50:] = -1
        y = tu.rms(x)
        self.assertAlmostEqual(y[0], 1)
        self.assertAlmostEqual(y[1], 1)

    def test_rms_tensor(self):
        x = torch.ones(100)
        x[50:] = -1
        self.assertAlmostEqual(tu.rms(x), 1)

    def test_rms_multichannel_tensor(self):
        x = torch.ones((2, 100))
        x[:, 50:] = -1
        y = tu.rms(x)
        self.assertAlmostEqual(y[0], 1)
        self.assertAlmostEqual(y[1], 1)

    def test_snr(self):
        x = np.ones(10) * 10
        noise = np.ones(10) * 1
        self.assertAlmostEqual(tu.snr(x, noise), 20)

    def test_snr_tensor(self):
        x = torch.ones(10) * 10
        noise = torch.ones(10) * 1
        self.assertAlmostEqual(tu.snr(x, noise), 20)

    def test_fade_sides_1d(self):
        x = np.ones(200)
        x = tu.fade_sides(x)
        self.assertEqual(x[0], 0)
        self.assertEqual(x[-1], 0)

    def test_fade_sides_3d(self):
        x = np.ones((1, 2, 200))
        x = tu.fade_sides(x)
        self.assertEqual(x[0, 0, 0], 0)
        self.assertEqual(x[0, 0, -1], 0)

    def test_tensor_fade_sides_1d(self):
        x = torch.ones(200)
        x = tu.fade_sides(x)
        self.assertEqual(x[0], 0)
        self.assertEqual(x[-1], 0)

    def test_tensor_fade_sides_3d(self):
        x = np.ones((1, 2, 200))
        x = tu.fade_sides(x)
        self.assertEqual(x[0, 0, 0], 0)
        self.assertEqual(x[0, 0, -1], 0)

    def test_trim(self):
        x = np.zeros((1, 1, 5 * 160))
        y = tu.trim(x, 160, 2)
        self.assertEqual(y.shape, (1, 1, 2 * 160))

    def test_tensor_trim(self):
        x = torch.zeros((1, 1, 5 * 160))
        y = tu.trim(x, 160, 2)
        self.assertEqual(y.shape, (1, 1, 2 * 160))


class TestGeneric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_pack_many(self):
        xs = [1, 2, 3]
        ys = [4, 5, 6]
        zss = tu.pack_many(xs, ys)
        self.assertEqual(zss, [(1, 4), (2, 5), (3, 6)])


if __name__ == "__main__":
    unittest.main()
