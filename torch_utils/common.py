from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler
from typing import Any, Callable, List, Tuple, Type, Union
from torchaudio.functional import resample
import torch.nn.functional as F
from random import randrange
from pathlib import Path
from torch import Tensor
import soundfile as sf
import numpy as np
import torchaudio
import torch
import h5py
import yaml

# export list
__all__ = [
    # generic utilities
    "DotDict",
    "Config",
    "get_np_or_torch",
    "pack_many",
    "repeat_test",
    # pytorch utilities
    "get_device",
    "to_numpy",
    "split_complex",
    "set_device",
]


# = = = = generic utilities


class DotDict(dict):
    """
    A dict that allows dot notation
    for accessing to the elements
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self) -> "DotDict":
        return DotDict(super().copy())


class Config:
    def __init__(self, config_path: Path) -> None:
        """
        A class to handle yaml configuration files.
        Inside a configuration many sections can be defined,
        with parameters within them.

        Parameters
        ----------
        config_path : Path
            Path to the yaml configuration file
        """
        # loading the configuration
        err_msg = "only YAML configurations are supported"
        assert config_path.suffix == ".yml", err_msg
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(
        self,
        section: str,
        parameter: str,
        _type: Type = str,
        default: Any = None,
    ) -> Any:
        """
        Gets a parameter from the configuration.

        Parameters
        ----------
        section : str
            Configuration section
        parameter : str
            Name of the parameter
        _type : Type, optional
            Type of the parameter, by default str
        default : Any, optional
            Default if the parameter does not exist, by default None

        Returns
        -------
        Fetched parameter
        """
        cfg = self.config

        try:
            # getting the section
            sec = cfg[section]
            # getting the parameter
            param = _type(sec[parameter])
        except KeyError:
            return default

        return param


def get_np_or_torch(x: Union[np.ndarray, Tensor]):
    """
    Returns numpy or torch modules depending on the input

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Input

    Returns
    -------
    Module
        numpy or torch
    """
    if isinstance(x, Tensor):
        return torch
    else:
        torch.iscomplex = torch.is_complex
        torch.hanning = torch.hann_window
        return np


def pack_many(*xss: List[List]) -> List[Tuple]:
    """
    Packs many Lists in one.

    Parameters
    ----------
    xss : List[List]

    Returns
    -------
    List[Tuple]
        Packed list
    """
    return list(zip(*xss))


# fmt: off
def repeat_test(times: int): 
    """
    Decorator to repeat a test n times.

    Parameters
    ----------
    times : int
        Number of repetitions
    """
    def repeatHelper(f):
        def callHelper(*args):
            for _ in range(0, times):
                f(*args)
        return callHelper
    return repeatHelper 
# fmt: on


# = = = = pytorch utilities


def get_device() -> str:
    """
    Gets the first CUDA device available or CPU
    if no CUDA device is available.

    Returns
    -------
    str
        Device id
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x: Tensor) -> np.ndarray:
    """
    Converts a Tensor into a numpy array.

    Parameters
    ----------
    x : Tensor
        Original tensor

    Returns
    -------
    np.ndarray
        Converted np array
    """
    return x.cpu().detach().numpy()

# TODO: test
def split_complex(x:Tensor)->Tensor:
    """
    Splits a complex Tensor in a float 
    Tensor with the double of the channels.

    Parameters
    ----------
    x : Tensor
        Complex input of shape (B, C, T, F)

    Returns
    -------
    Tensor
        Float output of shape (B, 2*C, T, F)
    """
    x = torch.cat((x.real, x.imag), dim=1)
    return x


def set_device(device: str) -> None:
    """
    Sets the default pytorch tensor
    to 'torch.{device}.FloatTensor'

    if device == "auto" it's inferred from get_device()

    Parameters
    ----------
    device : str
        Name of the device or "auto"
    """
    if device == "cpu":
        torch.set_default_tensor_type(f"torch.FloatTensor")
    else:
        if device == "auto":
            device = get_device()
        torch.set_default_tensor_type(f"torch.{device}.FloatTensor")
