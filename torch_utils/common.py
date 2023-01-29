from typing import Any, Callable, List, Tuple, Type, TypeVar, Union, Dict
import torch.nn.functional as F
from functools import partial
from torch import nn, Tensor
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
    "pack_complex",
    "set_device",
    "auto_device",
    "load_model",
    "get_submodules",
    "get_gradients",
    "get_model_parameters",
    "quantize",
    "one_hot_quantization",
    "invert_one_hot",
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


def split_complex(x: Tensor) -> Tensor:
    """
    Splits a complex Tensor in a float
    Tensor with the double of the channels.

    Parameters
    ----------
    x : Tensor
        Complex input of shape (B, C, ...)

    Returns
    -------
    Tensor
        Float output of shape (B, 2*C, ...)
    """
    module = get_np_or_torch(x)
    x = module.cat((x.real, x.imag), 1)
    return x


def pack_complex(x: Tensor) -> Tensor:
    """
    Merges a 2*C channels float Tensor in a complex Tensor.

    Parameters
    ----------
    x : Tensor
        Float input of shape (B, 2*C, ...)

    Returns
    -------
    Tensor
        Complex output of shape (B, C, ...)
    """
    c = x.shape[1]
    x = x[:, : c // 2] + 1j * x[:, c // 2 :]
    return x


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


def load_model(
    model_path: Path,
    model_class: nn.Module,
    *model_args: List,
    **model_kwargs: Dict,
) -> nn.Module:
    """
    Creates a model from its last checkpoint.

    Parameters
    ----------
    model_path : Path
        Basepath to the model
    model_class : nn.Module
        Class of the model

    Returns
    -------
    nn.Module
        Loaded model
    """

    sort_key = lambda x: -int(x.name.split(".ckpt")[0].split("_")[1])
    checkpoints = model_path / "checkpoints"
    checkpoints = list(checkpoints.glob("*.ckpt"))
    checkpoints.sort(key=sort_key)
    config = model_path / "config.yml"
    model_state = torch.load(checkpoints[0], map_location=get_device())
    model_state = model_state["model_state"]
    model = model_class(config, *model_args, **model_kwargs)
    model.load_state_dict(model_state)
    model.eval()
    return model


def get_submodules(model: nn.Module) -> List[nn.Module]:
    """
    Gets all the submodules without children from a model.

    Parameters
    ----------
    model : nn.Module
        Target model

    Returns
    -------
    List[nn.Module]
        Model submodules
    """
    modules = [x for x in model.modules() if len(list(x.children())) == 0]
    return modules


def get_gradients(model: nn.Module) -> Tensor:
    """
    Gets a model gradients norm through its submodules.

    Parameters
    ----------
    model : nn.Module
        Target model

    Returns
    -------
    Tensor
        Model gradients
    """
    modules = get_submodules(model)

    valid = (
        lambda x, n: hasattr(x, n)
        and isinstance(getattr(x, n), Tensor)
        and getattr(x, n).grad is not None
    )

    _norm = lambda x: torch.linalg.norm(x).item()
    f = lambda n: [_norm(getattr(x, n).grad) if valid(x, n) else 0 for x in modules]
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    w_grad = f("weight")
    b_grad = f("bias")
    # weight normalization
    g_grad = f("weight_g")
    v_grad = f("weight_v")
    # GRU ~ ~ ~ ~ ~ ~ ~ ~
    max_gru = 4
    gru_w_ih = [f(f"weight_ih_l{i}") for i in range(max_gru)]
    gru_w_hh = [f(f"weight_hh_l{i}") for i in range(max_gru)]
    bias_w_ih = [f(f"bias_ih_l{i}") for i in range(max_gru)]
    bias_w_hh = [f(f"bias_hh_l{i}") for i in range(max_gru)]
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    zipped = zip(w_grad, b_grad, g_grad, v_grad, *gru_w_ih, *gru_w_hh, *bias_w_ih, *bias_w_hh)
    grad = [sum(xs) for xs in zipped]
    grad = torch.FloatTensor(grad)
    return grad


def get_model_parameters(model: nn.Module) -> int:
    """
    Gets the total number of parameters (trainable and untrainable).

    Parameters
    ----------
    model : nn.Module
        Target model

    Returns
    -------
    int
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def quantize(x: Tensor, steps: int, min: float = -1, max: float = 1) -> Tensor:
    """
    Quantize a real signal.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (B, ...)
    steps : int
        Number of quantization steps
    min : float, optional
        Minimum value of the input (mapped to 0), by default -1
    max : float, optional
        Maximum value of the input (mapped to steps-1), by default 1

    Returns
    -------
    Tensor
        Quantized tensor of shape (B, ...)
    """
    x = (x - min) / (max - min) * steps
    x = x.floor().clip(0, steps - 1).to(int)
    return x


def one_hot_quantization(x: Tensor, steps: int, min: float = -1, max: float = 1) -> Tensor:
    """
    Quantize a real signal and applies a one-hot vector transform.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (B, ...)
    steps : int
        Number of quantization steps
    min : float, optional
        Minimum value of the input (mapped to 0), by default -1
    max : float, optional
        Maximum value of the input (mapped to steps-1), by default 1

    Returns
    -------
    Tensor
        One-hot-quantized tensor of shape (B, steps, ...)
    """
    n = len(x.shape)
    dims = list(range(n))
    dims.insert(1, n)

    # quantization
    x = quantize(x, steps, min, max)

    # one-hot vector
    x = F.one_hot(x, steps)
    x = x.permute(dims)

    return x


def invert_one_hot(x: Tensor) -> Tensor:
    """
    Transforms a one-hot tensor to a integer label representation.

    Parameters
    ----------
    x : Tensor
        On-hot tensor of shape (B, steps, ...)

    Returns
    -------
    Tensor
        Label representation of shape (B, ...)
    """
    x = torch.argmax(x, dim=1)
    return x
