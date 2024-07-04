__all__ = [
    "OneOrPair",
    "TensorOrArray",
    "disable_cuda",
    "get_np_or_torch",
    "pack_many",
    "execute_with_probability",
    "to_numpy",
    "phase",
    "transpose",
    "factorize",
]

from typing import List, Sequence, Tuple, TypeVar, Union
from numpy import ndarray
from torch import Tensor
import numpy as np
import torch
import os


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


def disable_cuda():
    """Disables CUDA in PyTorch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


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


def factorize(n: int) -> Sequence[int]:
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
    Sequence[int]
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
