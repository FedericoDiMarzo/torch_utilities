from typing import Iterator, List, Dict, Tuple
from torch import autograd as AG
import torch.nn.functional as F
from torch import nn, Tensor
from pathlib import Path
import numpy as np
import torch
import yaml

# export list
__all__ = [
    # pytorch utilities
    "enable_anomaly_detection",
    "load_model",
    "load_checkpoints",
    "sort_checkpoints",
    "get_submodules",
    "compute_gradients",
    "get_gradients",
    "get_model_parameters",
    "freeze_model",
    "quantize",
    "one_hot_quantization",
    "invert_one_hot",
    "CosineScheduler",
]


from torch_utilities.common import get_device


def enable_anomaly_detection():
    """
    Enable PyTorch anomaly detection
    """
    from torch import autograd

    autograd.anomaly_mode.set_detect_anomaly(True)


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
    # loading from best checkpoint
    checkpoint_dir = model_path / "checkpoints"
    checkpoint_scores = checkpoint_dir / "ckpt.yml"
    checkpoint_scores = load_checkpoints(checkpoint_scores)
    best_checkpoint = checkpoint_dir / checkpoint_scores[0][0]
    config = model_path / "config.yml"
    model_state = torch.load(best_checkpoint, map_location=get_device())
    model_state = model_state["model_state"]

    # initializing the model
    model = model_class(config, *model_args, **model_kwargs)
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_checkpoints(score_file: Path) -> List[Tuple[str, float]]:
    """
    Loads the ckpt.yml content ordered by the score.

    Parameters
    ----------
    score_file : Path
        Path to ckpt.yml

    Returns
    -------
    List[Tuple[str, float]]
        List of (checkpoint name, checkpoint score)
    """
    # loading
    with open(score_file) as f:
        checkpoints = yaml.unsafe_load(f)

    # ordering
    checkpoints = sort_checkpoints(checkpoints)
    return checkpoints


def sort_checkpoints(checkpoints: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Sorts the checkpoint data structure based on the scores.

    Parameters
    ----------
    checkpoints : List[Tuple[str, float]]
        Checkpoint data structure, loaded with load_checkpoints.

    Returns
    -------
    List[Tuple[str, float]]
        Ordered checkpoints
    """
    _order = lambda t: -t[1]
    checkpoints = sorted(checkpoints, key=_order)
    return checkpoints


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
    zipped = zip(
        w_grad, b_grad, g_grad, v_grad, *gru_w_ih, *gru_w_hh, *bias_w_ih, *bias_w_hh
    )
    grad = [sum(xs) for xs in zipped]
    grad = torch.FloatTensor(grad)
    return grad


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


def freeze_model(model: nn.Module) -> None:
    """
    Freezed the weights of a target module.

    Parameters
    ----------
    model : nn.Module
        Model to freeze
    """
    for p in model.parameters():
        p.requires_grad = False


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


def one_hot_quantization(
    x: Tensor, steps: int, min: float = -1, max: float = 1
) -> Tensor:
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


class CosineScheduler:
    def __init__(
        self,
        start_value: float,
        final_value: float,
        total_epochs: int,
        iterations_per_epoch: int = 1,
        warmup_epochs: int = 0,
    ) -> None:
        """
        Implements a single cycle cosine scheduler.

        Parameters
        ----------
        start_value : float
            Initial value
        final_value : float
            Last value
        total_epochs : int
            Total number of epochs
        iterations_per_epoch : int, optional
            Iterations for each eopoch, by default 1
        warmup_epochs : int, optional
            Number of warmup epochs (linear rise to the start_value), by default 0
        """
        self.start_value = start_value
        self.final_value = final_value
        self.total_epochs = total_epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.warmup_epochs = warmup_epochs

        self.schedule = self._compute_schedule()
        self.current_index = 0

    def __iter__(self) -> Iterator[float]:
        return iter(self.schedule.tolist())

    def __next__(self) -> float:
        value = self.schedule[self.current_index]
        self.current_index += 1
        return value

    def reset(self) -> None:
        """
        Resets the scheduler state.
        """
        self.current_index = 0

    def _compute_schedule(self) -> np.ndarray:
        """
        Computes the full schedule.

        Returns
        -------
        np.ndarray
            Schedule array
        """
        warmup_schedule = np.array([])
        warmup_iters = self.warmup_epochs * self.iterations_per_epoch
        if self.warmup_epochs > 0:
            warmup_start = self.start_value / warmup_iters
            warmup_schedule = np.linspace(warmup_start, self.start_value, warmup_iters)

        iters_after_warmup = (
            self.total_epochs * self.iterations_per_epoch - warmup_iters
        )
        cycle_lengths = [iters_after_warmup]

        schedule_cycles = []

        # supporting one single cycle
        num_cycles = 1
        for i in range(num_cycles):
            iters = np.arange(cycle_lengths[i])
            schedule = np.array(
                [
                    self.final_value
                    + 0.5
                    * (self.start_value - self.final_value)
                    * (1 + np.cos(np.pi * i / (len(iters))))
                    for i in iters
                ]
            )
            schedule_cycles.append(schedule)

        schedule = np.concatenate((warmup_schedule, *schedule_cycles))
        schedule = schedule[: self.total_epochs * self.iterations_per_epoch]

        assert len(schedule) == self.total_epochs * self.iterations_per_epoch
        return schedule
