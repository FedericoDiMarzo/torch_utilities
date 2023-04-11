from typing import Any, List, Tuple, Type, TypeVar, Union, Dict
from numpy import ndarray
from pathlib import Path
from torch import Tensor
from ray import tune
import numpy as np
import torch
import yaml

# export list
__all__ = [
    # types
    "OneOrPair",
    "TensorOrArray",
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
]

# = = = = types

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

    def get_ray_tune_params(self, section: str = "ray_tune") -> Dict:
        """
        Gets the ray tune parameters from a configuration.
        The parameters should be written in the section specified
        and be divided by sampling method.

        E.g.
        ray_tune:
            uniform:
                param_0: [0, 5]
            randn:
                param_1: [0, 1]


        Parameters
        ----------
        section : str, optional
            The parent section where the parameter space is defined,
            by default "ray_tune"

        Returns
        -------
        Dict
            Dictionary containing the ray tune parameters
        """
        try:
            cfg = self.config[section]
        except KeyError:
            # no ray tune section
            return {}

        sampling_methods = cfg.keys()
        params = {}

        # adding each hyperparameters parsing the
        # values passed to the corrispondent tune
        # sampling function f
        for sm in sampling_methods:
            f = getattr(tune, sm)
            _parsetype = lambda x: x if type(x) in (list, tuple) else float(x)
            params = params | {p: f(*map(_parsetype, v)) for p, v in cfg[sm].items()}
        return params


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
