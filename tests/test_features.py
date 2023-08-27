from typing import Iterable, Tuple
from torch import Tensor
import numpy as np
import itertools
import unittest
import torch

import torch_utilities as tu
from torch_utilities.features import STFT, ISTFT


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestStftIstft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.params = self.get_params()

    def get_params(self) -> Iterable:
        sample_rate = (16000, 32000, 48000)
        hopsize_ms = (4, 8)
        overlap_ratio = (2, 4)
        win_oversamp = (1, 2)
        pack_niquist = (False, True)

        params = itertools.product(
            sample_rate,
            hopsize_ms,
            overlap_ratio,
            win_oversamp,
            pack_niquist,
        )
        return params

    def get_input(self, params: Tuple) -> Tensor:
        sample_rate = params[0]
        x = torch.randn(1, 1, sample_rate // 4)
        return x

    def get_instances(self, p: Tuple) -> Tuple[STFT, ISTFT]:
        (
            sample_rate,
            hopsize_ms,
            overlap_ratio,
            win_oversamp,
            pack_niquist,
        ) = p
        stft = STFT(sample_rate, hopsize_ms, overlap_ratio, win_oversamp, pack_niquist)
        istft = ISTFT(
            sample_rate, hopsize_ms, overlap_ratio, win_oversamp, pack_niquist
        )
        return stft, istft

    def test_inversion(self):
        for p in self.params:
            with self.subTest(p=p):
                stft, istft = self.get_instances(p)
                x = self.get_input(p)
                y = stft(x)
                x_hat = istft(y)
                x = x[..., : x_hat.shape[-1]]
                x_hat = x_hat[..., : x.shape[-1]]
                e_max = (x - x_hat).abs().max()
                self.assertLess(e_max, 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
