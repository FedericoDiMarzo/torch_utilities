from typing import Any, Callable, List, Tuple, Type, TypeVar, Union, Dict
from functools import partial
from numpy import ndarray
from pathlib import Path
from torch import Tensor
import numpy as np
import torch
import yaml

# export list
__all__ = [
    # types
    "OneOrPair",
    "TensorOrArray",
    # pytorch devices
    "get_device",
    "set_device",
    "auto_device",
    # generic utilities
    "DotDict",
    "Config",
    "get_np_or_torch",
    "pack_many",
    "repeat_test",
    "execute_with_probability",
    # tensor/array operations
    "to_numpy",
    "split_complex",
    "pack_complex",
    "phase",
    "transpose",
    # math utilities
    "factorize",
]

#  types

""" 
Generic variable
"""
T = TypeVar("T")

"""
Single or Pair of values of a certain type.
"""
OneOrPair = Union[T, Tuple[T, T]]

"""
Can be a torch Tensor or numpy ndarray.
"""
TensorOrArray = Union[Tensor, ndarray]


#  handling pytorch devices
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


def set_device(device: str, dtype: str = "Float") -> None:
    """
    Sets the default pytorch tensor
    to 'torch.{device}.FloatTensor'

    if device == "auto" it's inferred from get_device()

    Parameters
    ----------
    device : str
        Name of the device or "auto"
    dtype : str, optional
        Type of the tensor, by default "Float"
    """
    if device == "auto":
        device = get_device()
    if device == "cpu":
        torch.set_default_tensor_type(f"torch.{dtype}Tensor")
    else:
        torch.set_default_tensor_type(f"torch.{device}.{dtype}Tensor")


def auto_device(dtype: str = "Float") -> Callable:
    def _auto_device(f: Callable) -> Callable:
        """
        Decorator to set the pytorch device to auto.

        Parameters
        ----------
        f : Callable
            Function to decorate
        dtype : str, optional
            Type of the tensor, by default "Float"

        Returns
        -------
        Callable
            Decorated function
        """
        set_device("auto", dtype)
        return f

    return _auto_device


auto_device = partial(auto_device)

# generic utilities


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


def get_np_or_torch(x: TensorOrArray):
    """
    Returns numpy or torch modules depending on the input

    Parameters
    ----------
    x : TensorOrArray
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
        np.cat = np.concatenate
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


def execute_with_probability(prob: float) -> bool:
    """
    Returns True of false based on the
    random variable prob

    Parameters
    ----------
    prob : float
        Probability to execute, from 0 (never happen)
        to 1 (always happen)

    Returns
    -------
    bool
        True prob % of the times
    """
    return np.random.uniform() > prob


# tensor/array operations


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


def split_complex(x: TensorOrArray) -> TensorOrArray:
    """
    Splits a complex Tensor in a float
    Tensor with the double of the channels.

    Parameters
    ----------
    x : TensorOrArray
        Complex input of shape (B, C, ...)

    Returns
    -------
    TensorOrArray
        Float output of shape (B, 2*C, ...)
    """
    module = get_np_or_torch(x)
    x = module.cat((x.real, x.imag), 1)
    return x


def pack_complex(x: TensorOrArray) -> TensorOrArray:
    """
    Merges a 2*C channels float TensorOrArray in a complex TensorOrArray.

    Parameters
    ----------
    x : TensorOrArray
        Float input of shape (B, 2*C, ...)

    Returns
    -------
    TensorOrArray
        Complex output of shape (B, C, ...)
    """
    c = x.shape[1]
    x = x[:, : c // 2] + 1j * x[:, c // 2 :]
    return x


def phase(x: TensorOrArray) -> TensorOrArray:
    """
    Computes the phase of a complex signal x
    as in: phase(x) = exp{j*angle(x)}

    Parameters
    ----------
    x : TensorOrArray
        Input complex signal

    Returns
    -------
    TensorOrArray
        Elementwise phase
    """
    module = get_np_or_torch(x)
    y = module.exp(1j * module.angle(x))
    return y


def transpose(x: TensorOrArray, dim1: int, dim2: int) -> TensorOrArray:
    """
    Transpose a tensor over two dimensions.

    Parameters
    ----------
    x : TensorOrArray
        Input tensor
    dim1 : int, optional
        First dimension to transpose
    dim2 : int, optional
        Second dimension to transpose

    Returns
    -------
    TensorOrArray
        Transposed tensor
    """
    return x.transpose(dim1, dim2) if type(x) == Tensor else np.swapaxes(x, dim1, dim2)


# math utilities


def factorize(n: int) -> List[int]:
    """
    Factorize an integer number.
    Implementation based on
    https://stackoverflow.com/questions/16007204/factorizing-a-number-in-python

    Parameters
    ----------
    n : int
        Number to factorize

    Returns
    -------
    List[int]
        List of factors in increasing order
    """

    def get_factor(n):
        j = 2
        while n > 1:
            for i in range(j, int(np.sqrt(n + 0.05)) + 1):
                if n % i == 0:
                    n //= i
                    j = i
                    yield i
                    break
            else:
                if n > 1:
                    yield int(n)
                    break

    factors = list(get_factor(n))
    factors = sorted(factors)
    return factors
