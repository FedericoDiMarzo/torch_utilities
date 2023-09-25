import onnxruntime as ort
from typing import Tuple
from pathlib import Path
from torch import Tensor
import numpy as np


from torch_utilities.common import to_numpy, TensorOrArray
from torch_utilities.audio import MelFilterbank, stft

# export list
__all__ = ["DNSMOS"]


def _get_model_dir() -> Path:
    """
    Returns the directory containing the models.

    Returns
    -------
    Path
        Path to the directory
    """
    path = Path(__file__).parent / "models"
    return path


class DNSMOS:
    def __init__(self, onnx_exec_provider: str = "CPUExecutionProvider") -> None:
        """Microsoft DNSMOS model.

        Args:
            onnx_exec_provider (str, optional): Exectution provider used for ONNXRuntime. Defaults to "CPUExecutionProvider".
        """
        self.model_path = _get_model_dir() / "sig_bak_ovr.onnx"
        self.onnx_exec_provider = onnx_exec_provider
        self.sample_rate = 16000
        self.hopsize_ms = 10
        self.n_mels = 120
        self.n_freqs = int(self.sample_rate / 1000 * self.hopsize_ms * 2 + 1)
        self.inference_sess = self.get_inference_session()

        # transforms
        self.stft = lambda x: stft(
            x,
            sample_rate=self.sample_rate,
            hopsize_ms=self.hopsize_ms,
            win_oversamp=2,
        )
        self.mel_fb = MelFilterbank(
            sample_rate=self.sample_rate,
            n_freqs=self.n_freqs,
            n_mels=self.n_mels,
        )

    def get_inference_session(self) -> ort.InferenceSession:
        """
        Returns an inference session for the model.

        Returns
        -------
        ort.InferenceSession
            DNSMOS inference session
        """
        session = ort.InferenceSession(
            str(self.model_path),
            providers=[self.onnx_exec_provider],
        )
        return session

    def get_polyfit_val(
        self, ovr: float, sig: float, bak: float
    ) -> Tuple[float, float, float]:
        """
        Polynomial nonlinearity as in the original repo

        Parameters
        ----------
        ovr : float
            Ovr raw value
        sig : float
            Sig raw value
        bak : float
            Bak raw value

        Returns
        -------
        Tuple[float, float, float]
            Processed metrics
        """
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        ovr_poly = p_ovr(ovr)
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)

        return ovr_poly, sig_poly, bak_poly

    def __call__(self, x: TensorOrArray) -> Tuple[float, float, float]:
        """
        Compute ovr, sig and bak from an 16kHz input.

        Parameters
        ----------
        x : TensorOrArray
            Input of shape (C, T) sampled at 16kHz,
            if multichannel, the first channel only is considered

        Returns
        -------
        Tuple[float, float, float]
            (ovr, sig, bak)
        """
        assert len(x.shape) == 2, "the shape of x should be (C, T)"

        if isinstance(x, Tensor):
            x = to_numpy(x)
        x = x.astype("float32")

        # keeping the first channel
        x = x[:1]

        # computing the average over segments of 1s
        samples_selected = 144160
        n_frames = x.shape[1] // samples_selected
        ovr = []
        sig = []
        bak = []
        for i in range(n_frames):
            a = i * samples_selected
            b = (i + 1) * samples_selected
            x_sel = x[:, a:b]
            model_in = dict(input_1=x_sel)
            scores = self.inference_sess.run(None, model_in)[0][0]
            scores = self.get_polyfit_val(*scores)
            [container.append(s) for container, s in zip((sig, bak, ovr), scores)]
        ovr, sig, bak = [np.mean(container) for container in (ovr, sig, bak)]

        return ovr, sig, bak
