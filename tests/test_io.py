import numpy as np
import pytest
import torch

import torch_utilities as tu
from torch_utilities import TensorOrArray

# Local fixtures ===============================================================


@pytest.fixture(params=[True, False])
def remove_last(request) -> bool:
    """Remove last."""
    return request.param


# ===============================================================================================


class TestIO:
    def test_pack_audio_sequences(
        self, audio_file, workers, channels, sample_rate, module, remove_last
    ):
        length = 1.5
        tensor = module == torch
        n_files = 10

        files = [audio_file for _ in range(n_files)]
        itr = tu.pack_audio_sequences(
            xs=files,
            length=length,
            sample_rate=sample_rate,
            channels=channels,
            tensor=tensor,
            delete_last=remove_last,
            num_workers=workers,
        )
        for i, seq in enumerate(itr):
            assert seq.shape == (channels, int(sample_rate * length))
        n = i + 1 if remove_last else i
        assert n == int(n_files / length)
