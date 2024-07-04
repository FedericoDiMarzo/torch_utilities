from typing import Callable, Tuple
import numpy as np
import pytest
import torch

from torch_utilities import TensorOrArray
import torch_utilities as tu

# Local fixtures ===============================================================


@pytest.fixture(params=[160])
def n_freq(request) -> int:
    """The number of frequencies."""
    return request.param


@pytest.fixture(params=[32])
def n_mel(request) -> int:
    """The number of mel filters."""
    return request.param


@pytest.fixture
def mel_filterbank(sample_rate, n_freq, n_mel) -> tu.MelFilterbank:
    """Mel filterbank instance."""
    return tu.MelFilterbank(sample_rate, n_freq, n_mel)


@pytest.fixture
def mel_inverse_filterbank(sample_rate, n_freq, n_mel) -> tu.MelInverseFilterbank:
    """Mel filterbank instance."""
    return tu.MelInverseFilterbank(sample_rate, n_freq, n_mel)


def _input_common(module, channels, sample_rate, f) -> TensorOrArray:
    x = np.ones((1, channels, int(sample_rate * 0.1), f), dtype=complex)
    if module == torch:
        x = torch.from_numpy(x)
    return x


@pytest.fixture
def input_spec(module, channels, sample_rate, n_freq) -> TensorOrArray:
    """Input spectrogram."""
    return _input_common(module, channels, sample_rate, n_freq)


@pytest.fixture
def input_mel(module, channels, sample_rate, n_mel) -> TensorOrArray:
    """Input mel."""
    return _input_common(module, channels, sample_rate, n_mel)


@pytest.fixture
def rand_float(module) -> TensorOrArray:
    x = np.random.uniform(0, 1, 1)
    if module == torch:
        x = torch.tensor(x)
    return x


@pytest.fixture(params=[0, 10])
def trim_margin(request) -> int:
    """Trim margin."""
    return request.param


@pytest.fixture(params=[128])
def n_fft(request) -> int:
    """Number of FFT points."""
    return request.param


@pytest.fixture
def n_stft_freqs(n_fft) -> int:
    """Number of FFT frequencies."""
    return n_fft // 2 + 1


@pytest.fixture()
def hop_size(n_fft) -> int:
    """Hop size."""
    return n_fft // 2


@pytest.fixture(params=[True, False])
def use_complex(request) -> bool:
    """Use complex."""
    return request.param


@pytest.fixture
def stft_istft(sample_rate, n_fft, hop_size, use_complex) -> Tuple[Callable, Callable]:
    """STFT and ISTFT functions."""
    stft, istft = tu.get_stft_istft(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_size=hop_size,
        complex=use_complex,
    )

    return stft, istft


@pytest.fixture
def stft(stft_istft) -> Callable:
    """STFT function."""
    stft, _ = stft_istft
    return stft


@pytest.fixture
def istft(stft_istft) -> Callable:
    """ISTFT function."""
    _, istft = stft_istft
    return istft


# ==============================================================================


class TestSTFT:
    def test_stft_shape(self, stft, use_complex, n_stft_freqs):
        x = torch.zeros(1, 1, 16000)
        y = stft(x)
        assert y.shape[:2] == x.shape[:2]
        assert len(y.shape) == 4
        if use_complex:
            assert torch.is_complex(y)
            assert y.shape[-1] == n_stft_freqs
        else:
            assert not torch.is_complex(y)
            assert y.shape[-1] == 2 * n_stft_freqs

    def test_istft_shape(self, istft, use_complex, n_stft_freqs):
        x = torch.zeros(1, 1, 10, 2 * n_stft_freqs)
        if use_complex:
            x = x[..., :n_stft_freqs] + 1j
        y = istft(x)
        assert not torch.is_complex(y)
        assert y.shape[:2] == (1, 1)
        assert len(y.shape) == 3


class TestMelFilterbank:
    def test_to_mel_shape(self, mel_filterbank, input_spec, n_mel):
        x = input_spec
        y = mel_filterbank(x)
        assert x.shape[:-1] == y.shape[:-1]
        assert y.shape[-1] == n_mel

    def test_to_freq_shape(self, mel_inverse_filterbank, input_mel, n_freq):
        x = input_mel
        y = mel_inverse_filterbank(x)
        assert x.shape[:-1] == y.shape[:-1]
        assert y.shape[-1] == n_freq


class TestAudio:
    def test_db(self, rand_float):
        x = rand_float
        y = tu.db(x)
        x_hat = tu.invert_db(y)
        torch.testing.assert_close(x, x_hat)

    def test_power(self, module):
        x = module.ones((2, 100))
        x[50:] = -1
        torch.testing.assert_close(tu.power(x), module.ones(2) * 100.0)

    def test_energy(self, module):
        x = module.ones((2, 100))
        x[50:] = -1
        torch.testing.assert_close(tu.energy(x), module.ones(2) * 1.0)

    def test_rms(self, module):
        x = module.ones((2, 100))
        x[50:] = -1
        torch.testing.assert_close(tu.rms(x), module.ones(2) * 1.0)

    def test_fade_sides_2d(self, module):
        x = module.ones((2, 200))
        x = tu.fade_sides(x)
        for i in range(2):
            zero = module.zeros(1)
            torch.testing.assert_close(x[i, 0], zero[0])
            torch.testing.assert_close(x[i, -1], zero[0])

    def test_trim(self, module):
        x = module.zeros((1, 1, 5 * 160))
        y = tu.random_trim(x, 160, 2)
        assert y.shape == (1, 1, 2 * 160)

    def test_trim_silence(self, module, trim_margin):
        x = module.zeros(100)
        x[30:40] = 0.1
        y = tu.trim_silence(x, margin=trim_margin)
        assert y.shape[-1] == (10 + 2 * trim_margin)

    def test_interleave(self, module):
        dimensions = 3
        freqs = 10
        x_size = [freqs if i == (dimensions - 1) else 1 for i in range(dimensions)]

        if module == np:
            x = np.random.normal(size=x_size)
        else:
            x = torch.randn(x_size)

        xs = [x for _ in range(dimensions)]
        y = tu.interleave(*xs)
        y_dim = x_size
        y_dim[-1] *= dimensions
        assert y.shape == tuple(y_dim)
        for i, z in enumerate(xs):
            e = module.abs(y[..., i::dimensions] - z).max()
            assert e < 1e-6

    def test_trim_as_shortest(self, module):
        x = module.ones((1, 100, 1))
        y = module.ones((1, 50, 1))
        x, y = tu.trim_as_shortest(x, y, dim=1)
        assert x.shape == (1, 50, 1)
        assert y.shape == (1, 50, 1)
