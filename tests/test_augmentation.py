import numpy as np
import pytest
import torch

import torch_utilities as tu
from torch_utilities import TensorOrArray
import torch_utilities.augmentation as aug


# Local fixtures ===============================================================


@pytest.fixture(params=[-12, 0, 12])
def snr(request) -> int:
    """The signal-to-noise ratio."""
    return request.param


# ==============================================================================


class TestAugmentation:
    def test_shuffle(self):
        x = torch.arange(1000)
        y = aug.shuffle(x)
        assert x.shape == y.shape
        assert torch.any(x.not_equal(y))

    def test_add_noise(self, snr):
        x = torch.ones((1, 1, 100))
        n = torch.ones_like(x)
        y, actual_snr = aug.add_noise(x, n, (snr, snr))
        assert x.max().equal(y.max())
        assert actual_snr - tu.invert_db(snr) < 1e-6
        assert len(actual_snr.shape) == 1
        assert actual_snr.shape[0] == y.shape[0]

    # def test_scale(self):
    #     scale_set = (-12, 0, 12)
    #     for scale in scale_set:
    #         with self.subTest(snr=scale):
    #             lin_scale = aug.invert_db(scale)
    #             x = torch.ones((1, 1, 100))
    #             y, actual_scaling = aug.random_scaling(x, (scale, scale))
    #             self.assertAlmostEqual(y.max().item(), lin_scale)
    #             self.assertLess(actual_scaling - invert_db(scale), 1e-6)
    #             self.assertTrue(len(actual_scaling.shape), 1)
    #             self.assertTrue(actual_scaling.shape[0], y.shape[0])

    # def test_overdrive(self):
    #     x = torch.ones((1, 1, 100))
    #     y = aug.random_overdrive(x)
    #     self.assertAlmostEqual(y.max().item(), 1)

    # def test_dc_removal(self):
    #     x = torch.ones((2, 3, 5))
    #     y = aug.dc_removal(x)
    #     m = y.mean()
    #     self.assertLess(m, 1e-6)
