from functools import reduce
from operator import mul
from typing import List
import numpy as np
import pytest
import torch

import torch_utilities as tu

# Local fixtures ===============================================================


# expected factorizations
@pytest.fixture(params=[[3, 7, 11], [3, 17], [2, 2, 2, 7]])
def expected_factorization(request) -> List[int]:
    return request.param


# ==============================================================================


class TestUtilities:
    def test_pack_many(self):
        xs = [1, 2, 3]
        ys = [4, 5, 6]
        zss = tu.pack_many(xs, ys)
        assert zss == [(1, 4), (2, 5), (3, 6)]

    def test_phase(self, module):
        phase = np.array([0, 1, 0.2])
        phase = np.exp(1j * phase)
        if module == torch:
            phase = torch.from_numpy(phase)
        x = module.ones_like(phase)
        x = x * phase
        phase_hat = tu.phase(x)
        torch.testing.assert_close(phase, phase_hat)

    def test_factorize(self, expected_factorization):
        exp = expected_factorization
        x = reduce(mul, exp)
        y = tu.factorize(x)
        assert y == exp
