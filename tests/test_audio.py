import unittest
from pathimport import set_module_root
import numpy as np
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as tu
from torch_utils import repeat_test, set_auto_device

torch.manual_seed(984)
np.random.seed(901)
set_auto_device()


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


if __name__ == "__main__":
    unittest.main()
