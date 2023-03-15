from resampy import resample as resample_np
from torchaudio.functional import resample
from pathimport import set_module_root
from typing import Iterator, List, Tuple, Union
from multiprocess import Pool
from pathlib import Path
from torch import Tensor
import soundfile as sf
import numpy as np
import torchaudio
import torch

set_module_root(".")
from torch_utilities.common import get_device, to_numpy, TensorOrArray
from torch_utilities.audio import MelFilterbank, stft

# export list
__all__ = ["DNSMOS"]

# TODO: continue here


def _get_model_dir() -> Path:
    path = Path(__file__).parent.parent / "resources" / "models"
    return path


class DNSMOS:
    def __init__(self) -> None:
        self.model_path = _get_model_dir() / "sig_bak_ovr.onnx"
        self.sample_rate = 16000
        self.hopsize_ms = 10
        self.n_mels = 120
        self.mel_fb = MelFilterbank(
            sample_rate=self.sample_rate,
            n_freqs=int(self.hopsize_ms + 1),
            n_mels=self.n_mels,
        )

    def compute_features(self, x: TensorOrArray) -> TensorOrArray:
        """
        Preprocess a time domain signal into the input representation
        of DNSMOS.

        Parameters
        ----------
        x : TensorOrArray
            Signal of shape (B, C, T)

        Returns
        -------
        TensorOrArray
            Features of shape (B, C, T', F)
        """
        x = stft(
            x,
            sample_rate=self.sample_rate,
            hopsize_ms=self.hopsize_ms,
            win_len_ms=(2 * self.hopsize_ms),
            win_oversamp=1,
        )
