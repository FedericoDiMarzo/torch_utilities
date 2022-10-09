import unittest
from pathlib import Path
from pathimport import set_module_root
from torch import Tensor
import numpy as np
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as TU


def get_test_data_dir():
    return Path(__file__).parent / "test_data"


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.config_path = get_test_data_dir() / "test.yaml"
        self.config = TU.Config(self.config_path)

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
        y_true = 12.34
        y_test = self.config.get("section2", "param4", float)
        self.assertAlmostEqual(y_true, y_test)

    def test_default_none(self):
        y_true = None
        y_test = self.config.get("section1", "param10")
        self.assertEqual(y_true, y_test)

    def test_default(self):
        y_true = "default"
        y_test = self.config.get("section1", "param10", default="default", type=int)
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
        x = TU.load_audio(self.mono_file)[0]
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(np.ndarray, type(x))

    def test_stereo_read(self):
        x = TU.load_audio(self.stereo_file)[0]
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(np.ndarray, type(x))

    def test_tensor_mono_read(self):
        x = TU.load_audio(self.mono_file, tensor=True)[0]
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(Tensor, type(x))

    def test_tensor_stereo_read(self):
        x = TU.load_audio(self.stereo_file, tensor=True)[0]
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(Tensor, type(x))

    def test_mono_save(self):
        x = np.zeros((1, self.tmp_sr))
        TU.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_stereo_save(self):
        x = np.zeros((2, self.tmp_sr))
        TU.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_tensor_mono_save(self):
        x = torch.zeros((1, self.tmp_sr))
        TU.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()

    def test_tensor_stereo_save(self):
        x = torch.zeros((2, self.tmp_sr))
        TU.save_audio(self.tmp_file, x, self.tmp_sr)
        self.delete_tmp()


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
            x_stft = TU.stft(
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
            x_stft = TU.stft(
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
        x_istft = TU.istft(
            x,
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
            frame_oversampling=4,
        )
        self.assertEqual(len(x_istft.shape), 1)

    def test_multich_istft(self):
        x = np.stack([self.x_np_stft] * 4)
        x_istft = TU.istft(
            x,
            sample_rate=self.sample_rate,
            framesize_ms=self.framesize_ms,
            frame_oversampling=4,
        )
        self.assertEqual(len(x_istft.shape), 2)
        self.assertEqual(x_istft.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
