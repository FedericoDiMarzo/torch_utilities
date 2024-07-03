__all__ = [
    "enable_anomaly_detection",
    "get_submodules",
    "compute_gradients",
    "get_num_parameters",
    "freeze_model",
]

from torch import autograd as AG
from torch import nn, Tensor
from typing import Sequence
import torch


def enable_anomaly_detection():
    """
    Enable PyTorch anomaly detection
    """
    from torch import autograd

    autograd.anomaly_mode.set_detect_anomaly(True)


def get_submodules(model: nn.Module) -> Sequence[nn.Module]:
    """
    Gets all the submodules without children from a model.

    Parameters
    ----------
    model : nn.Module
        Target model

    Returns
    -------
    Sequence[nn.Module]
        Model submodules
    """
    modules = [x for x in model.modules() if len(list(x.children())) == 0]
    return modules


def compute_gradients(x: Tensor, y: Tensor, keep_graph: bool = True) -> Tensor:
    """
    Computes the matrix-vector product between the Jacobian calculated from the
    function with y as output and x as input and a vector of ones of the same
    shape as y.

    Parameters
    ----------
    x : Tensor
        Input of the target function, must require grad
    y : Tensor
        Output of the target function
    keep_graph : bool, optional
        If True the result can be used in further autograd graphs, it can then be used
        for higher degree derivative optimization (i.e. as an derivative optimization term in a loss
        such in https://arxiv.org/abs/1704.00028), by default True

    Returns
    -------
    Tensor
        matrix-vector product between the Jacobian
    """
    ones = torch.ones_like(y)
    grad = AG.grad(y, x, ones, retain_graph=keep_graph, create_graph=keep_graph)[0]
    return grad


def get_num_parameters(model: nn.Module) -> int:
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


def freeze_model(model: nn.Module):
    """
    Freezed the weights of a target module.

    Parameters
    ----------
    model : nn.Module
        Model to freeze
    """
    for p in model.parameters():
        p.requires_grad = False
