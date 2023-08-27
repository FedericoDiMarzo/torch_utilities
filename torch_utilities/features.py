import torch.nn.functional as F
from torch import Tensor


from torch_utilities.common import split_complex, pack_complex
from torch_utilities.audio import stft, istft

# export list
__all__ = [
    "STFT",
    "ISTFT",
]


class STFT:
    def __init__(
        self,
        sample_rate: int,
        hopsize_ms: float,
        overlap_ratio: int = 2,
        win_oversamp: int = 1,
        pack_niquist: bool = True,
    ) -> None:
        """
        Apply a STFT to the input batch.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the signal
        hopsize_ms : int
            STFT hopsize in ms
        overlap_ratio : int, optional
            Overlap ratio between two frames
        win_oversamp : int, optional
            Zero padding oversampling applied to the window,
            by default 1 equals to no oversampling
        pack_niquist : bool, optional
            If True the niquist frequency is encoded as the imaginary part of
            the DC bin, by default True
        """
        self.sample_rate = sample_rate
        self.hopsize_ms = hopsize_ms
        self.overlap_ratio = overlap_ratio
        self.win_oversamp = win_oversamp
        self.pack_niquist = pack_niquist

    def __call__(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Signal of shape (B, C, T)

        Returns
        -------
        Tensor
            STFT of shape (B, 2*C, T', F)
        """
        batches, channels = x.shape[:2]
        x = x.flatten(0, 1)
        x = stft(
            x,
            sample_rate=self.sample_rate,
            hopsize_ms=self.hopsize_ms,
            win_len_ms=self.hopsize_ms * self.overlap_ratio,
            win_oversamp=self.win_oversamp,
        )
        x = x.unflatten(0, (batches, channels))
        if self.pack_niquist:
            # putting the niquist as imag of 0-th bin
            x[..., 0] += 1j * x[..., -1].real
            x = x[..., :-1]
        x = split_complex(x)
        return x


class ISTFT:
    def __init__(
        self,
        sample_rate: int,
        hopsize_ms: float,
        overlap_ratio: int = 2,
        win_oversamp: int = 1,
        unpack_niquist: bool = True,
    ) -> None:
        """
        Apply a ISTFT to the input batch.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the signal
        hopsize_ms : int
            STFT hopsize in ms
        overlap_ratio : int, optional
            Overlap ratio between two frames
        win_oversamp : int, optional
            Zero padding oversampling applied to the window,
            by default 1 equals to no oversampling
        unpack_niquist : bool, optional
            If True the niquist frequency is decoded as the imaginary part of
            the DC bin, by default True
        """
        self.sample_rate = sample_rate
        self.hopsize_ms = hopsize_ms
        self.sample_rate = sample_rate
        self.hopsize_ms = hopsize_ms
        self.overlap_ratio = overlap_ratio
        self.win_oversamp = win_oversamp
        self.unpack_niquist = unpack_niquist

    def __call__(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Signal of shape (B, 2*C, T, F)

        Returns
        -------
        Tensor
            ISTFT of shape (B, C, T')
        """
        x = pack_complex(x)

        if self.unpack_niquist:
            # putting back the niquist at its place
            x = F.pad(x, (0, 1))
            x[..., -1] = x[..., 0].imag + 0j
            x[..., 0] = x[..., 0].real + 0j

        batches, channels = x.shape[:2]
        x = x.flatten(0, 1)

        x = istft(
            x,
            sample_rate=self.sample_rate,
            hopsize_ms=self.hopsize_ms,
            win_len_ms=self.hopsize_ms * self.overlap_ratio,
            win_oversamp=self.win_oversamp,
        )
        x = x.unflatten(0, (batches, channels))
        return x
