from pathimport import set_module_root
import numpy as np
import unittest
import torch

set_module_root("../torch_utils")
import torch_utilities as tu
import torch_utilities.augmentation as aug


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        pass

    def test_shuffle(self):
        torch.manual_seed(0)
        x = torch.arange(1000)
        y = aug.shuffle(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.any(x.not_equal(y)))

    def test_add_noise(self):
        snr_set = (-12, 0, 12)
        for snr in snr_set:
            with self.subTest(snr=snr):
                x = torch.ones((1, 1, 100))
                n = torch.ones_like(x)
                y = aug.add_noise(x, n, (snr, snr))
                self.assertTrue(x.max().equal(y.max()))

    def test_scale(self):
        scale_set = (-12, 0, 12)
        for scale in scale_set:
            with self.subTest(snr=scale):
                lin_scale = aug.invert_db(scale)
                x = torch.ones((1, 1, 100))
                y = aug.scale(x, (scale, scale))
                self.assertAlmostEqual(y.max().item(), lin_scale)

    def test_overdrive(self):
        x = torch.ones((1, 1, 100))
        y = aug.overdrive(x)
        self.assertAlmostEqual(y.max().item(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
