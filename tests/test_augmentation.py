import pytest
import torch

import torch_utilities.augmentation as aug
import torch_utilities as tu


# Local fixtures ===============================================================


@pytest.fixture(params=[-12, 0, 7])
def snr(request) -> int:
    """The signal-to-noise ratio."""
    return request.param


# scale
@pytest.fixture(params=[-3, 0])
def scale(request) -> int:
    """The scaling factor."""
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

    def test_scale(self, scale):
        batch_size = 4
        lin_scale = tu.invert_db(scale)
        x = torch.ones((batch_size, 1, 100))
        y, actual_scaling = aug.random_scaling(x, (scale, scale))
        assert y.shape == x.shape
        assert actual_scaling.shape == (x.shape[0],)
        torch.testing.assert_close(y.max().item(), lin_scale)
        torch.testing.assert_close(actual_scaling, torch.ones(batch_size) * lin_scale)

    def test_overdrive(self):
        x = torch.ones((4, 1, 100))
        y = aug.random_overdrive(x)
        torch.testing.assert_close(y.max().item(), 1.0)

    def test_dc_removal(self):
        x = torch.ones((2, 3, 5))
        y = aug.dc_removal(x)
        m = y.mean()
        assert m < 1e-6
