from torch import Tensor
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


# ==============================================================================


class TestIO:
    def test_load_audio(self, audio_file, channels, sample_rate, module):
        is_tensor = module == torch
        expected_type = Tensor if is_tensor else np.ndarray
        x, x_sr = tu.load_audio(audio_file, sample_rate, is_tensor)
        assert len(x.shape) == 2
        assert x.shape[0] == channels
        assert type(x) == expected_type
        assert x_sr == sample_rate

    def test_save_audio(self, channels, sample_rate, module, session_tmp_dir):
        out_path = session_tmp_dir / "save_audio_out.wav"
        out_path.unlink(missing_ok=True)
        x = module.zeros((channels, sample_rate // 16))
        tu.save_audio(out_path, x, sample_rate)
        assert out_path.exists()

    def test_load_audio_parallel(
        self, audio_file, channels, sample_rate, module, workers
    ):
        n_files = 2 * workers
        is_tensor = module == torch
        expected_type = Tensor if is_tensor else np.ndarray
        filepaths = [audio_file for _ in range(n_files)]
        xs = tu.load_audio_parallel(
            filepaths, sample_rate, is_tensor, num_workers=workers
        )
        assert len(xs) == n_files
        for x in xs:
            assert type(x) == expected_type
            assert x.shape[0] == channels

    def test_load_audio_parallel_itr(
        self, audio_file, channels, sample_rate, module, workers
    ):
        n_files = 2 * workers
        is_tensor = module == torch
        expected_type = Tensor if is_tensor else np.ndarray
        filepaths = [audio_file for _ in range(n_files)]
        xs = tu.load_audio_parallel_itr(
            filepaths, sample_rate, is_tensor, num_workers=workers
        )
        for i, x in enumerate(xs):
            assert type(x) == expected_type
            assert x.shape[0] == channels
        assert i + 1 == n_files

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
