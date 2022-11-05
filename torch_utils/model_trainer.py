from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch.utils.data import DataLoader
from pathimport import set_module_root
from torch import optim, nn, Tensor
from collections import deque
from loguru import logger
import torch_utils as tu
from pathlib import Path
import numpy as np
from abc import ABC
import torch
import abc

set_module_root(".", prefix=True)
from torch_utils.common import DotDict


class ModelTrainer(ABC):
    def __init__(
        self,
        model_path: Path,
        model: nn.Module,
        train_ds: tu.HDF5Dataset,
        valid_ds: tu.HDF5Dataset,
        optimizer: optim.Optimizer,
        losses: List[Callable],
        losses_names: Optional[List[str]] = None,
    ) -> None:
        """
        Abstract class to structure a model training.

        The directory pointed by model_path should
        have the following structure:

        .
        ├── config.yml
        ├── checkpoints  // optional, from previous trainings
        └── logs         // optional, from previous trainings

        Use the section [training] of config.yml to store
        additional parameters

        Parameters
        ----------
        model_path : Path
            Path to the model directory
        model : nn.Module
            Model to train
        train_ds : tu.HDF5Dataset
            Training dataset
        train_ds : tu.HDF5Dataset
            Validation dataset
        optimizer : optim.Optimizer
            Optimizer
        losses : List[Callable]
            List of losses to be computed
        losses_names : Optional[List[str]]
            Names of the losses, by default [loss0, loss1, ...]

        config.yml [training] parameters
        ----------
        max_epochs : int, optional
            Max number of epochs, by default 100
        losses_weights : List[float], optional
            Per-loss gains, by default all ones
        log_every : int, optional
            Frequency of the logs (over the dataset iterations),
            by default 100
        """
        # explicit attributes
        self.model_path = model_path
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.optimizer = optimizer
        self.losses = losses
        self.losses_names = losses_names or self._default_losses_names()

        # configuration attributes
        self.config_path = self.model_path / "config.yml"
        self.config = self._parse_config(self.config_path)
        self.losses_weight = self.from_config("losses_weights", list, np.ones(len(self.losses)))
        self.log_every = self.from_config("log_every", int, 100)
        self.max_epochs = self.from_config("max_epochs", int, 100)

        # other dirs
        self.checkpoints_dir = model_path / "checkpoints"
        self.logs_dir = model_path / "logs_dir"
        [d.make_dir(exist_ok=True) for d in (self.checkpoints_dir, self.logs_dir)]

        # model and running_losses setup
        self.net = self.load_model()
        self.running_losses = None
        self._reset_running_losses()

        # extra stuff
        self.save_buffer = deque([], maxlen=5)

    def start_training(self) -> None:
        """
        Trains a model.
        """
        # TODO: recover from previous training

        for epoch in range(self.max_epochs):
            logger.info(f"epoch [{epoch}/{self.max_epochs}]")

            # training
            self.model.train()
            for i, data in enumerate(self.train_ds):
                self.train_step(data)
                if i % self.log_every == 0 and i != 0:
                    self._log_losses(is_training=True, steps=self.log_every)
                    self._reset_running_losses()
                #
            self._reset_running_losses()
            self.save_model(epoch)

            # validation
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.valid_ds):
                    self.valid_step(data)
                self._log_losses(is_training=False, steps=i)
                self._reset_running_losses()

    def train_step(self, data: List[Tensor]) -> None:
        """
        Single step of a training loop.

        Parameters
        ----------
        data : List[Tensor]
            Inputs to the model
        """
        self.optimizer.zero_grad()
        net_outputs = self.net(*data)
        _losses = self.apply_losses(net_outputs)
        _losses = self._apply_losses_weights(_losses)
        total_loss = sum(_losses)
        total_loss.backward()
        self.optimizer.step()
        # for logging
        self._update_running_losses(_losses)

    def valid_step(self, data: List[Tensor]) -> None:
        """
        Single step of a validation loop.

        Parameters
        ----------
        data : List[Tensor]
            Inputs to the model
        """
        net_outputs = self.net(*data)
        _losses = self.apply_losses(net_outputs)
        _losses = self._apply_losses_weights(_losses)
        self._update_running_losses(_losses)

    def load_model(self) -> nn.Module:
        """
        Loads a model from its class and configuration file.

        Returns
        -------
        nn.Module
            Loaded model
        """
        # TODO: load checkpoint if it exists
        m = self.model(self.config)
        return m

    def _apply_losses_weights(
        self,
        losses: List[Union[Tensor, float]],
    ) -> List[Union[Tensor, float]]:
        """
        Weights the losses by losses_weights.

        Parameters
        ----------
        losses : List[Union[Tensor, float]]
            List of losses to weight

        Returns
        -------
        Tensor
            Weighted losses
        """
        return [loss * w for loss, w in zip(losses, self.losses_weight)]

    def _reset_running_losses(self) -> None:
        """
        Resets the state of the running_losses.
        """
        self.running_losses = np.zeros(len(self.losses))

    def _log_losses(self, is_training: bool, steps: int) -> None:
        """
        Logs running_losses.

        Parameters
        ----------
        is_training : bool
            Flag to separate train/valid logging
        """
        scope = "[TRAIN LOSSES]" if is_training else "[VALID LOSSES]"
        logger.info(scope)
        for loss, name in zip(self.running_losses, self.losses_names):
            logger.info(f"{name}: {loss}")

        self._reset_running_losses()

    def _update_running_losses(self, losses: List[Tensor]) -> None:
        """
        Updates the running losses.

        Parameters
        ----------
        losses : List[Tensor]
            Current losses
        """
        losses = [loss.item() for loss in losses]
        self.running_losses = [l0 + l1 for l0, l1 in zip(self.running_losses, losses)]

    @abc.abstractmethod
    def apply_losses(self, net_outputs: List[Tensor]) -> List[Tensor]:
        """
        Use the new_outputs to calculate the losses and
        return them.

        Parameters
        ----------
        net_outputs : List[Tensor]
            Network outputs

        Returns
        -------
        List[Tensor]
            List of computed losses (not weighted)
        """
        pass

    def _parse_config(config: Path) -> tu.DotDict:
        """
        Parse the configuration file
        to read the training section.

        Parameters
        ----------
        config : Path
            Path to the config file

        Returns
        -------
        Dict
            Dictionary of the training section
        """
        config = tu.Config(config)
        _config = tu.DotDict(
            learning_rate=config.get("training", "learning_rate", float, 0.001),
        )
        return _config

    def from_config(self, param: str, _type: Type, default: Any = None) -> Any:
        """
        gets a parameter from the training configuration.

        Parameters
        ----------
        param : str
            Name of the parameter
        _type : Type
            Parameter type
        default : Any, optional
            Default value, by default None

        Returns
        -------
        Any
            Value of the parameter
        """
        return self.config.get("training", param, _type, default)

    def save_model(self, epoch: int) -> None:
        """
        Saves the model in the checkpoints folder.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        # saving
        name = self.checkpoints_dir / f"checkpoint_{epoch}.ckpt"
        torch.save(self.net.state_dict(), self)

        # removing old checkpoints
        self.save_buffer.append(name)
        sb = self.save_buffer
        checkpoints = self.checkpoints_dir.glob("*.ckpt")
        targets = filter(lambda x: x.name not in sb, checkpoints)
        [x.unlink() for x in targets]
