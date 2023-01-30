from pathimport import set_module_root
from contextlib import suppress
from torch import Tensor
import numpy as np
import itertools
import unittest
import torch

set_module_root("../torch_utils")
import torch_utilities as tu


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestSTFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        module = (np, torch)  # 0
        channels = (1, 4)  # 1
        sample_rate = (8000, 16000, 24000, 48000)  # 2
        hopsize_ms = (8, 10)  # 3
        win_oversamp = (1, 2)  # 5
        self.params = itertools.product(
            module,
            channels,
            sample_rate,
            hopsize_ms,
            win_oversamp,
        )
        self.params = [(*p[:4], 2 * p[3], p[4]) for p in self.params]

    def test_stft(self):
        for p in self.params:
            mod = p[0]
            with self.subTest(p=p):
                x = mod.ones((p[1], p[2] * 1))
                y = tu.stft(x, p[2], p[3], "hann", p[4], p[5])
                bins = int(p[2] * p[4] * p[5] / 2000 + 1)
                self.assertEqual(y.shape[-1], bins)

    def test_istft(self):
        for p in self.params:
            mod = p[0]
            with self.subTest(p=p):
                bins = int(p[2] * p[4] * p[5] / 2000 + 1)
                x = mod.ones((p[1], int(p[2] * 0.05), bins)) + 0j
                y = tu.istft(x, p[2], p[3], "hann", p[4], p[5])

    def test_inversion(self):
        for p in self.params:
            mod = p[0]
            with self.subTest(p=p):
                if mod == np:
                    x = np.random.normal(size=(p[1], p[2] * 1))
                else:
                    x = torch.randn((p[1], p[2] * 1))

                y = tu.stft(x, p[2], p[3], "hann", p[4], p[5])
                x_hat = tu.istft(y, p[2], p[3], "hann", p[4], p[5])
                x = x[0, : x_hat.shape[1]]
                x_hat = x_hat[0, : x.shape[0]]
                err = mod.abs(x - x_hat).max()
                with suppress(Exception):
                    err = err.item()
                self.assertAlmostEqual(err, 0, places=5)


class TestMelFilterbank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        module = (np, torch)
        channels = (1, 4)
        sample_rate = (8000, 16000, 24000, 48000)
        n_freq = (160, 320, 640, 1280)
        n_mel = (8, 16, 32)

        self.params = itertools.product(
            module,
            channels,
            sample_rate,
            n_freq,
            n_mel,
        )

    def test_to_mel(self):
        for (module, channels, sample_rate, n_freq, n_mel) in self.params:
            with self.subTest(
                module=module,
                channels=channels,
                sample_rate=sample_rate,
                n_freq=n_freq,
                n_mel=n_mel,
            ):
                filterbank = tu.MelFilterbank(sample_rate, n_freq, n_mel)
                x = np.ones((1, channels, int(sample_rate * 0.1), n_freq), dtype=complex)
                x = x if module == np else torch.from_numpy(x).to(tu.get_device())
                y = filterbank(x)
                self.assertEqual(y.shape, (1, channels, int(sample_rate * 0.1), n_mel))
                self.assertEqual(type(y), np.ndarray if module == np else Tensor)

    def test_to_freq(self):
        for (module, channels, sample_rate, n_freq, n_mel) in self.params:
            with self.subTest(
                module=module,
                channels=channels,
                sample_rate=sample_rate,
                n_freq=n_freq,
                n_mel=n_mel,
            ):
                filterbank = tu.MelInverseFilterbank(sample_rate, n_freq, n_mel)
                x = np.ones((1, channels, int(sample_rate * 0.1), n_mel), dtype=complex)
                x = x if module == np else torch.from_numpy(x).to(tu.get_device())
                y = filterbank(x)
                self.assertEqual(y.shape, (1, channels, int(sample_rate * 0.1), n_freq))
                self.assertEqual(type(y), np.ndarray if module == np else Tensor)


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.modules = (np, torch)

    def test_db(self):
        eps_list = (1e-12, 1e-3, 10)
        for eps in eps_list:
            with self.subTest(eps=eps):
                x = np.random.uniform(0, 1, 100)
                y = tu.db(x, eps)
                x_hat = tu.invert_db(y, eps)
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
        y = tu.random_trim(x, 160, 2)
        self.assertEqual(y.shape, (1, 1, 2 * 160))

    def test_tensor_trim(self):
        x = torch.zeros((1, 1, 5 * 160))
        y = tu.random_trim(x, 160, 2)
        self.assertEqual(y.shape, (1, 1, 2 * 160))

    def test_trim_silence(self):
        margins = (0, 10)
        values = (1, 0.1, 100)
        params = itertools.product(self.modules, margins, values)
        for module, marg, val in params:
            x = module.zeros(100)
            x[30:40] = 0.1
            with self.subTest(module=module, marg=marg, val=val):
                y = tu.trim_silence(x, margin=marg)
                self.assertEqual(y.shape[-1], 10 + 2 * marg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
