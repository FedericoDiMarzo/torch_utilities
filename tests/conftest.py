from pathlib import Path
import numpy as np
import random
import pytest
import torch

import torch_utilities as tu

# These fixtures are applied to all tests in the test suite. =====================================


@pytest.fixture(autouse=True, scope="session")
def reset_seeds():
    """Reset seeds for reproducibility."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@pytest.fixture(autouse=True, scope="session")
def disable_gradient():
    """Disable gradient computation."""
    torch.set_grad_enabled(False)


@pytest.fixture(autouse=True, scope="session")
def disable_gpu():
    """Disable GPU for tests."""
    tu.disable_cuda()


# ===============================================================================================


@pytest.fixture(params=[np, torch], scope="session")
def module(request):
    """Fixture to parametrize the module to test."""
    return request.param


@pytest.fixture(params=[1, 4], scope="session")
def channels(request):
    """Fixture to parametrize the number of channels."""
    return request.param


@pytest.fixture(params=[16000], scope="session")
def sample_rate(request):
    """Fixture to parametrize the sample rate."""
    return request.param


@pytest.fixture(params=[1, 2], scope="session")
def workers(request):
    return request.param


@pytest.fixture(scope="session")
def session_tmp_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def audio_file(session_tmp_dir, channels, sample_rate) -> Path:
    """One second audio file."""
    x = np.random.uniform(-1, 1, (channels, sample_rate))
    filename = session_tmp_dir / "audio.wav"
    tu.save_audio(filename, x, sample_rate)
    return filename
