from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathimport import set_module_root
from torch import optim, nn, Tensor
import matplotlib.pyplot as plt
from contextlib import suppress
from collections import deque
from loguru import logger
import torch_utils as tu
from pathlib import Path
import numpy as np
from abc import ABC
import warnings
import torch
import abc

set_module_root(".", prefix=True)
from torch_utils.common import DotDict

__all__ = ["ModelTrainer"]


class ModelTrainer(ABC):
    def __init__(
        self,
        model_path: Path,
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer_class: optim.Optimizer,
        losses: List[Callable],
        losses_names: Optional[List[str]] = None,
        overfit_mode: bool = False,
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
        train_dl : DataLoader
            Training DataLoader
        train_dl : DataLoader
            Validation DataLoader
        optimizer_class : optim.Optimizer
            Optimizer class
        losses : List[Callable]
            List of losses to be computed
        losses_names : Optional[List[str]]
            Names of the losses, by default ["loss0", "loss1", ...]
        overfit_mode : bool, optional
            Enables overfit mode, by default False


        config.yml [training] parameters
        ----------
        max_epochs : int, optional
            Max number of epochs, by default 100
        learning_rate : float, optional
            Optimizer learning rate, by default 0.001
        weight_decay : float, optional
            Optimizer weight_decay, by default 0
        losses_weights : List[float], optional
            Per-loss gains, by default all ones
        log_every : int, optional
            Frequency of the logs (over the dataset iterations),
            by default 100
        """
        # explicit attributes
        self.model_path = model_path
        self.model = model
        self.train_ds = train_dl
        self.valid_ds = valid_dl
        self.losses = losses
        self.losses_names = losses_names or self._default_losses_names()
        self.optimizer_class = optimizer_class
        self.overfit_mode = overfit_mode

        # configuration attributes
        self.config_path = self.model_path / "config.yml"
        self.config = tu.Config(self.config_path)
        self.learning_rate = self._from_config("learning_rate", float, 0.001)
        self.weight_decay = self._from_config("weight_decay", float, 0)
        self.log_every = self._from_config("log_every", int, 100)
        self.max_epochs = self._from_config("max_epochs", int, 100)
        self.losses_weight = self._from_config(
            "losses_weights", np.array, np.ones(len(self.losses))
        )

        # other dirs
        self.checkpoints_dir = model_path / "checkpoints"
        self.logs_dir = model_path / "logs_dir"
        [d.mkdir(exist_ok=True) for d in (self.checkpoints_dir, self.logs_dir)]

        # model and running_losses setup
        self.start_epoch = 0
        self.net = self.load_model()
        self.optim_state = None
        self.optimizer = self._setup_optimizer()
        self.running_losses = None
        self._reset_running_losses()

        # extra stuff
        self.save_buffer = deque([], maxlen=5)  # TODO: best saving mechanism
        self.log_writer = SummaryWriter(self.logs_dir)
        self.figsize = (8, 6)

    # = = = = = = = = = = = = = = = = = = = = = =
    #             Training loop
    # = = = = = = = = = = = = = = = = = = = = = =
    def start_training(self) -> None:
        """
        Trains a model.
        """
        logger.info(f"device selected: {tu.get_device()}")
        logger.info(f"model: {self.model_path.name}")
        logger.info(f"parameters: {self._get_model_parameters() / 1e3} K")
        if self.overfit_mode:
            logger.info("overfit mode on")

        # logging
        logger.info("saving the model graph")
        self._log_graph()
        logger.info("saving the text of the config.yaml")
        self._log_yaml()

        logger.info("starting training")  # - = - § >>
        self.on_train_begin()
        for epoch in range(self.start_epoch, self.max_epochs):
            # TODO: on_epoch_begin()
            logger.info(f"epoch [{epoch}/{self.max_epochs}]")

            # training
            self.net.train()
            for i, data in enumerate(self.train_ds):
                if self._is_loading_batches(self.train_ds):
                    data = data[0]
                with torch.no_grad():
                    data = self.apply_transforms(data)
                self.train_step(data)
                if i % self.log_every == 0 and i != 0:
                    self._log_losses(is_training=True, steps=self.log_every, epoch=epoch)
                    self._reset_running_losses()
                #
            self.save_model(epoch)
            self._log_gradients(epoch)
            self._log_losses(is_training=True, steps=(i % self.log_every) + 1, epoch=epoch)
            self._reset_running_losses()

            with torch.no_grad():
                _dl = lambda t: self.train_ds if t else self.valid_ds
                _log_data = lambda t: [x.to(tu.get_device()) for x in _dl(t).dataset[[0, 1]]]
                self.tensorboard_logs(_log_data(True), epoch=epoch, is_training=True)
                self._log_outs(epoch)

                if not self.overfit_mode:
                    # validation
                    self.net.eval()
                    for i, data in enumerate(self.valid_ds):
                        if self._is_loading_batches(self.valid_ds):
                            data = data[0]
                        data = self.apply_transforms(data)
                        self.valid_step(data)
                    self._log_losses(is_training=False, steps=i + 1, epoch=epoch)
                    self._reset_running_losses()
                    self.tensorboard_logs(_log_data(False), epoch=epoch, is_training=False)

        logger.info("training complete")  # - = - § >>
        self.on_train_end()

    def train_step(self, data: List[Tensor]) -> None:
        """
        Single step of a training loop.

        Parameters
        ----------
        data : List[Tensor]
            Inputs to the model
        """
        self.on_train_step_begin()
        data = [x.to(tu.get_device()) for x in data]
        self.optimizer.zero_grad()
        net_outputs = self.net(*data)
        _losses = self.apply_losses(data, net_outputs)
        _losses = self._apply_losses_weights(_losses)
        total_loss = sum(_losses)
        total_loss.backward()
        self.optimizer.step()
        # for logging
        self._update_running_losses(_losses)
        self.on_train_step_end()

    def valid_step(self, data: List[Tensor]) -> None:
        """
        Single step of a validation loop.

        Parameters
        ----------
        data : List[Tensor]
            Inputs to the model
        """
        self.on_valid_step_begin()
        data = [x.to(tu.get_device()) for x in data]
        net_outputs = self.net(*data)
        _losses = self.apply_losses(data, net_outputs)
        _losses = self._apply_losses_weights(_losses)
        self._update_running_losses(_losses)
        self.on_valid_step_end()

    # = = = = = = = = = = = = = = = = = = = = = =
    #            Handling Losses
    # = = = = = = = = = = = = = = = = = = = = = =
    @abc.abstractmethod
    def apply_losses(self, net_ins: List[Tensor], net_outs: List[Tensor]) -> List[Tensor]:
        """
        Use the new_outputs to calculate the losses and
        return them.

        Parameters
        ----------
        net_ins : List[Tensor]
            Network inputs
        net_outs : List[Tensor]
            Network outputs

        Returns
        -------
        List[Tensor]
            List of computed losses (not weighted)
        """
        pass

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

    def _reset_running_losses(self) -> None:
        """
        Resets the state of the running_losses.
        """
        self.running_losses = np.zeros(len(self.losses))

    # = = = = = = = = = = = = = = = = = = = = = =
    #             Input Features
    # = = = = = = = = = = = = = = = = = = = = = =
    @abc.abstractmethod
    def apply_transforms(self, net_ins: List[Tensor]) -> List[Tensor]:
        """
        Apply tranforms to the inputs.

        Parameters
        ----------
        net_ins : List[Tensor]
            Network inputs

        Returns
        -------
        List[Tensor]
            Transformed input
        """
        pass

    # = = = = = = = = = = = = = = = = = = = = = =
    #               Callbacks
    # = = = = = = = = = = = = = = = = = = = = = =
    def on_train_begin(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_train_step_begin(self) -> None:
        pass

    def on_train_step_end(self) -> None:
        pass

    def on_valid_step_begin(self) -> None:
        pass

    def on_valid_step_end(self) -> None:
        pass

    # = = = = = = = = = = = = = = = = = = = = = =
    #             Model loading
    # = = = = = = = = = = = = = = = = = = = = = =
    def load_model(self) -> nn.Module:
        """
        Loads a model from its class and configuration file.

        Returns
        -------
        nn.Module
            Loaded model
        """
        m = self.model(self.config_path)

        # load checkpoint if it exists
        if self._prev_train_exists():
            epoch, model_state, optim_state = self._load_checkpoint()
            self.start_epoch = epoch
            self.optim_state = optim_state
            m.load_state_dict(model_state)

        return m

    def _prev_train_exists(self) -> bool:
        """
        Checks if a previous training exists.

        Returns
        -------
        bool
            True if a previous training exists
        """
        if not self.checkpoints_dir.exists():
            return False
        checkpoints = self._get_checkpoints()
        return len(checkpoints) > 0

    def _get_checkpoints(self) -> List[Path]:
        """
        Gets the checkpoints paths.

        Returns
        -------
        List[Path]
            Checkpoints paths
        """
        checkpoints = list(self.checkpoints_dir.glob("*.ckpt"))
        return checkpoints

    def _load_checkpoint(self) -> Tuple[int, Dict, Dict]:
        """
        Loads the last checkpoint.

        Returns
        -------
        Tuple[int, Dict, Dict]
            (epoch, model_state, optim_state)
        """
        # choosing the last checkpoint
        checkpoints = self._get_checkpoints()
        chkp_idx = [int(x.name.split("_")[1].split(".ckpt")[0]) for x in checkpoints]
        i = np.argmax(chkp_idx)
        chosen = checkpoints[i]

        # loading and parsing
        chosen = DotDict(torch.load(chosen))
        epoch = chosen.epoch
        model_state = chosen.model_state
        optim_state = chosen.optim_state
        return epoch, model_state, optim_state

    # = = = = = = = = = = = = = = = = = = = = = =
    #             Model saving
    # = = = = = = = = = = = = = = = = = = = = = =
    def save_model(self, epoch: int) -> None:
        """
        Saves the model in the checkpoints folder.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        # saving
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{epoch}.ckpt"
        torch.save(
            dict(
                model_state=self.net.state_dict(),
                optim_state=self.optimizer.state_dict(),
                epoch=epoch,
            ),
            checkpoint_path,
        )
        self._push_new_checkpoint(checkpoint_path.name)

    def _push_new_checkpoint(self, checkpoint_name: str) -> None:
        """
        Pushes a new checkpoint into the save_buffer.

        Parameters
        ----------
        checkpoint_name : str
            Name of the checkpoint
        """

        # removing old checkpoints
        self.save_buffer.append(checkpoint_name)
        sb = self.save_buffer
        checkpoints = self.checkpoints_dir.glob("*.ckpt")
        targets = filter(lambda x: x.name not in sb, checkpoints)
        [x.unlink() for x in targets]

    # = = = = = = = = = = = = = = = = = = = = = =
    #           Optimizer loading
    # = = = = = = = = = = = = = = = = = = = = = =
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Sets up the optimizer to the model.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance
        """
        try:
            optim = self.optimizer_class(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        except TypeError:
            # no weight decay
            optim = self.optimizer_class(
                self.net.parameters(),
                lr=self.learning_rate,
            )

        if self.optim_state is not None:
            with suppress(RuntimeError):
                # ignore errors for optimizer mismatches
                optim.load_state_dict(self.optim_state)
        return optim

    # = = = = = = = = = = = = = = = = = = = = = =
    #                Logging
    # = = = = = = = = = = = = = = = = = = = = = =
    def get_tensorboard_writer(self) -> SummaryWriter:
        """
        Getter to the tensorboard log writer

        Returns
        -------
        SummaryWriter
            tensorboard log writer
        """
        return self.log_writer

    @abc.abstractclassmethod
    def tensorboard_logs(self, net_ins: List[Tensor], epoch: int, is_training: bool) -> None:
        """
        Additional tensorboard logging.

        Parameters
        ----------
        net_ins : List[Tensor]
            Network raw inputs (no transforms)
        epoch : int
            Current epoch
        is_training : bool
            Flag to separate train/valid logging
        """
        pass

    def _get_dummy_input(self, is_training: bool) -> List[Tensor]:
        """
        Returns an input from the validation dataset

        Parameters
        -------
        is_training : bool
            Flag to separate train/valid logging

        Returns
        -------
        List[Tensor]
            Validation input selection
        """
        ds = self.train_ds if is_training else self.valid_ds
        x = [x.to(tu.get_device()) for x in ds.dataset[[0, 1]]]
        x = self.apply_transforms(x)
        return x

    def _log_losses(self, is_training: bool, steps: int, epoch: int) -> None:
        """
        Logs running_losses.

        Parameters
        ----------
        is_training : bool
            Flag to separate train/valid logging
        steps : int
            Iterations before this function was called
        epoch : int
            Current epoch
        """
        tag_suffix = "train" if is_training else "valid"
        losses_names = self.losses_names + ["total"]
        losses = self._apply_losses_weights(self.running_losses)
        total_loss = sum(losses)
        losses = self.running_losses + [total_loss]
        for loss, name in zip(losses, losses_names):
            loss /= steps
            logger.info(f"{name}: {loss}")
            self.log_writer.add_scalar(f"{name}_{tag_suffix}", loss, global_step=epoch)

        self._reset_running_losses()

    def _default_losses_names(self) -> List[str]:
        """
        Default loss names

        Returns
        -------
        List[str]
            ["loss0", "loss1", ...]
        """
        return [f"loss{i}" for i in range(len(self.losses))]

    def _log_gradients(self, epoch: int) -> None:
        """
        Saves a plot of the gradient norm in tensorboard.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        grad = tu.get_gradients(self.net)
        plt.figure(figsize=self.figsize)
        plt.bar(range(grad.shape[0]), grad)
        plt.title("model gradient")
        plt.grid()
        plt.xlabel("submodule index")
        plt.ylabel("gradient norm")
        fig = plt.gcf()
        self.log_writer.add_figure("gradient", fig, epoch)
        plt.close()

    def _log_outs(self, epoch: int) -> None:
        """
        Saves a plot of the outputs norm in tensorboard.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        net_ins = self._get_dummy_input(is_training=True)
        net_outs = self.net(*net_ins)
        norm = [torch.linalg.norm(x[0]).item() for x in net_outs]
        norm = Tensor(norm).cpu()
        plt.figure(figsize=self.figsize)
        plt.bar(range(norm.shape[0]), norm)
        plt.title("model outputs")
        plt.grid()
        plt.xlabel("output index")
        plt.ylabel("output norm")
        fig = plt.gcf()
        self.log_writer.add_figure("outputs", fig, epoch)
        plt.close()

    def _log_graph(self) -> None:
        """
        Saves the model graph.
        """
        x = self._get_dummy_input(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.log_writer.add_graph(self.net, x)

    def _log_yaml(self) -> None:
        """
        Saves the yaml text on Tensorboard
        """
        with open(self.config_path) as f:
            txt = f.read()
        self.log_writer.add_text("config.yaml", txt, 0)

    # = = = = = = = = = = = = = = = = = = = = = =
    #                Utilities
    # = = = = = = = = = = = = = = = = = = = = = =

    def _is_loading_batches(self, dl: DataLoader) -> bool:
        """
        Checks if a DataLoader is loading one item at the time (False)
        or multiple items (True).

        Parameters
        ----------
        dl : DataLoader
            Target DataLoader

        Returns
        -------
        bool
            True if multiple indices are passed to the DataLoader
        """
        samp = dl.sampler
        idx = iter(samp).__next__()
        return isinstance(idx, list)

    def _get_model_parameters(self) -> int:
        """
        Gets the total model parameters.

        Returns
        -------
        int
            Number of net parameters
        """
        params = tu.get_model_parameters(self.net)
        return params

    def _from_config(self, param: str, _type: Type, default: Any = None) -> Any:
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
