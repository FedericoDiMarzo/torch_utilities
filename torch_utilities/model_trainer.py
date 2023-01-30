from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch.utils.tensorboard import SummaryWriter
from contextlib import suppress, nullcontext
from torch.utils.data import DataLoader
from pathimport import set_module_root
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch import nn, Tensor
from loguru import logger
import torch_utilities as tu
from pathlib import Path
import numpy as np
from abc import ABC
import warnings
import torch
import abc

set_module_root(".")
from torch_utilities.common import DotDict

__all__ = ["ModelTrainer"]


class ModelTrainer(ABC):
    def __init__(
        self,
        model_path: Path,
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer_class: Type[Optimizer],
        losses: List[Callable],
        net_ins_indices: Optional[List[int]] = None,
        losses_names: Optional[List[str]] = None,
        save_buffer_maxlen: int = 5,
        gradient_clip_value: Optional[float] = None,
        enable_profiling: bool = False,
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
        optimizer_class : Type[Optimizer]
            Optimizer class
        losses : List[Callable]
            List of losses to be computed
        net_ins_indices : Optional[List[int]]
            Used to filter and reorder the List[Tensor] obtained
            from the dataloader, by default no filter.
            E.g.
                dataloader out = [T0, T1, T2]
                net_ins_indices = [2, 1]
                net_ins = [T2, T1]
        losses_names : Optional[List[str]]
            Names of the losses, by default ["loss0", "loss1", ...]
        save_buffer_maxlen : int, optional
            N best checkpoints saved (based on a lower total loss),
            by default 5
        gradient_clip_value : Optional[float]
            Maximum gradient update value, by default no gradient clipping
        enable_profiling : bool, optional
            If True runs one full run of the train dataset and
            logs to tensorboard the profiling information, by default False


        config.yml [training] parameters
        ----------
        max_epochs : int, optional
            Max number of epochs, by default 100
        learning_rate : float, optional
            Optimizer learning rate, by default 0.001
        overfit_mode : bool, optional
            Enables overfit mode, by default False
        weight_decay : float, optional
            Optimizer weight_decay, by default 0
        losses_weights : List[float], optional
            Per-loss gains, by default all ones
        log_every : int, optional
            Frequency of the logs (over the dataset iterations),
            by default 100
        """
        self.device = tu.get_device()

        # explicit attributes
        self.model_path = model_path
        self.model = model
        self.train_ds = train_dl
        self.valid_ds = valid_dl
        self.losses = losses
        self.net_ins_indices = net_ins_indices
        self.losses_names = losses_names or self._default_losses_names()
        self.optimizer_class = optimizer_class
        self.gradient_clip_value = gradient_clip_value
        self.enable_profiling = enable_profiling

        # configuration attributes
        self.config_path = self.model_path / "config.yml"
        self.config = tu.Config(self.config_path)
        self.learning_rate = self._from_config("learning_rate", float, 0.001)
        self.weight_decay = self._from_config("weight_decay", float, 0)
        self.log_every = self._from_config("log_every", int, 100)
        self.max_epochs = self._from_config("max_epochs", int, 100)
        self.overfit_mode = self._from_config("overfit_mode", bool, False)
        self.losses_weights = self._from_config(
            "losses_weights", np.array, np.ones(len(self.losses))
        )

        # other dirs
        self.checkpoints_dir = model_path / "checkpoints"
        self.logs_dir = model_path / "logs"
        [d.mkdir(exist_ok=True) for d in (self.checkpoints_dir, self.logs_dir)]

        # model and running_losses setup
        self.start_epoch = 0
        self.net = self.load_model()
        self.optim_state = None
        self.optimizer = self._setup_optimizer()
        self.running_losses = None
        self.running_losses_steps = 0
        self._reset_running_losses()

        # extra stuff
        self.save_buffer = []
        self.save_buffer_maxlen = save_buffer_maxlen
        self.log_writer = SummaryWriter(self.logs_dir)
        self.figsize = (8, 6)
        self.dummy_input_train = self._get_dummy_data(True)
        self.dummy_input_valid = self._get_dummy_data(False)
        self.last_total_loss = 1e10  # used to select the best checkpoints
        self.profiler = self._get_profiler()  # null context manager if enable_profiling==False

    # = = = = = = = = = = = = = = = = = = = = = =
    #             Training loop
    # = = = = = = = = = = = = = = = = = = = = = =
    def start_training(self) -> None:
        """
        Trains a model.
        """
        logger.info(f"device selected: {self.device}")
        logger.info(f"model: {self.model_path.name}")
        logger.info(f"parameters: {self._get_model_parameters() / 1e3} K")
        if self.overfit_mode:
            logger.info("overfit mode on")
        if self.enable_profiling:
            logger.info("profiler on: stopping after one epoch")

        # logging
        logger.info("saving the model graph")
        self._log_graph()
        logger.info("saving the text of config.yaml")
        self._log_yaml()
        msg = "starting profilation" if self.enable_profiling else "starting training"

        logger.info(msg)  # - = - § >>
        self._start_profiling()  # <- - profiler on
        self.on_train_begin()
        for epoch in range(self.start_epoch, self.max_epochs):
            self.on_epoch_begin(epoch)
            logger.info(f"epoch [{epoch+1}/{self.max_epochs}]")

            # training
            self.net.train()
            for i, data in enumerate(self.train_ds):
                data = self._remove_extra_dim(data)
                self.train_step(data, epoch)
                if i % self.log_every == 0 and i != 0:
                    self._log_losses(is_training=True, epoch=epoch)
                    self._reset_running_losses()
                #
            self._log_gradients(epoch)
            self._log_losses(is_training=True, epoch=epoch)
            self._reset_running_losses()

            with torch.no_grad():
                _log_data = lambda t: self.dummy_input_train if t else self.dummy_input_valid
                self.tensorboard_logs(_log_data(True), epoch=epoch, is_training=True)
                self._log_outs(epoch)

                self._stop_profiling()  # <- - profiler off

                # leaving the training  ~ ~
                if self.enable_profiling:
                    logger.info(f"{self.model_path.name} profilation complete")
                    return
                # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

                if not self.overfit_mode:
                    # validation
                    self.net.eval()
                    for i, data in enumerate(self.valid_ds):
                        data = self._remove_extra_dim(data)
                        self.valid_step(data, epoch)
                    self._log_losses(is_training=False, epoch=epoch)
                    self._reset_running_losses()
                    self.tensorboard_logs(_log_data(False), epoch=epoch, is_training=False)

            self.save_model(epoch)
            self.on_epoch_end(epoch)

        logger.info(f"{self.model_path.name} training complete")  # - = - § >>
        self.on_train_end()

    def train_step(self, data: List[Tensor], epoch: int) -> None:
        """
        Single step of a training loop.

        Parameters
        ----------
        data : List[Tensor]
            Inputs to the model
        epoch : int
            Current epoch
        """
        self.on_train_step_begin(epoch)
        data = [x.to(self.device) for x in data]
        net_ins = self._get_filtered_input(data)
        self.optimizer.zero_grad()
        net_outs = self.net(*net_ins)
        _losses = self.apply_losses(data, net_outs)
        _losses = self._apply_losses_weights(_losses)
        total_loss = sum(_losses)
        total_loss.backward()
        # gradient clipping ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        if self.gradient_clip_value is not None:
            nn.utils.clip_grad_value_(
                self.net.parameters(),
                clip_value=self.gradient_clip_value,
            )
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.optimizer.step()
        # for logging
        self._update_running_losses(_losses)
        self.on_train_step_end(epoch)

        # profiler update
        if self.enable_profiling:
            self.profiler.step()

    def valid_step(self, data: List[Tensor], epoch) -> None:
        """
        Single step of a validation loop.

        Parameters
        ----------
        data : List[Tensor]
            Inputs to the model
        epoch : int
            Current epoch
        """
        self.on_valid_step_begin(epoch)
        data = [x.to(self.device) for x in data]
        net_ins = [data[i] for i in self.net_ins_indices]
        net_outs = self.net(*net_ins)
        _losses = self.apply_losses(data, net_outs)
        _losses = self._apply_losses_weights(_losses)
        self._update_running_losses(_losses)
        self.on_valid_step_end(epoch)

    # = = = = = = = = = = = = = = = = = = = = = =
    #            Handling Losses
    # = = = = = = = = = = = = = = = = = = = = = =
    @abc.abstractmethod
    def apply_losses(self, data: List[Tensor], net_outs: List[Tensor]) -> List[Tensor]:
        """
        Use the new_outputs to calculate the losses and
        return them.

        Parameters
        ----------
        data : List[Tensor]
            List of Tensors loaded by the dataloader
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
        return [loss * w for loss, w in zip(losses, self.losses_weights)]

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
        self.running_losses_steps += 1

    def _reset_running_losses(self) -> None:
        """
        Resets the state of the running_losses.
        """
        self.running_losses = np.zeros(len(self.losses))
        self.running_losses_steps = 0

    # = = = = = = = = = = = = = = = = = = = = = =
    #               Callbacks
    # = = = = = = = = = = = = = = = = = = = = = =
    def on_train_begin(self: int) -> None:
        pass

    def on_train_end(self: int) -> None:
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def on_train_step_begin(self, epoch: int) -> None:
        pass

    def on_train_step_end(self, epoch: int) -> None:
        pass

    def on_valid_step_begin(self, epoch: int) -> None:
        pass

    def on_valid_step_end(self, epoch: int) -> None:
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
        chosen = DotDict(torch.load(chosen, map_location=self.device))
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
        # sorting by the lower loss
        _order = lambda t: t[1]
        self.save_buffer.append((checkpoint_name, float(self.last_total_loss)))
        self.save_buffer = sorted(self.save_buffer, key=_order)
        self.save_buffer = self.save_buffer[: self.save_buffer_maxlen + 1]

        self._delete_worse_checkpoints()

    def _delete_worse_checkpoints(self) -> None:
        """
        Deletes the checkpoints that are not in the save_buffer.
        """
        sb = [t[0] for t in self.save_buffer]
        checkpoints = self.checkpoints_dir.glob("*.ckpt")
        targets = filter(lambda x: x.name not in sb, checkpoints)
        [x.unlink() for x in targets]

    # = = = = = = = = = = = = = = = = = = = = = =
    #           Optimizer loading
    # = = = = = = = = = = = = = = = = = = = = = =
    def _setup_optimizer(self) -> Optimizer:
        """
        Sets up the optimizer to the model.

        Returns
        -------
        Optimizer
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
    def tensorboard_logs(self, raw_data: List[Tensor], epoch: int, is_training: bool) -> None:
        """
        Additional tensorboard logging.

        Parameters
        ----------
        raw_data : List[Tensor]
            Dataset input
        epoch : int
            Current epoch
        is_training : bool
            Flag to separate train/valid logging
        """
        pass

    def _get_dummy_data(self, is_training: bool) -> List[Tensor]:
        """
        Returns an input from the dataset

        Parameters
        -------
        is_training : bool
            Flag to separate train/valid logging

        Returns
        -------
        List[Tensor]
            Dataset input
        """
        ds = self.train_ds if is_training else self.valid_ds
        data = [x.to(self.device) for x in ds.dataset[[0, 1]]]
        return data

    def _log_losses(self, is_training: bool, epoch: int) -> None:
        """
        Logs running_losses.

        Parameters
        ----------
        is_training : bool
            Flag to separate train/valid logging
        epoch : int
            Current epoch
        """
        if self.running_losses_steps == 0:
            # nothing to log
            return

        tag_suffix = "train" if is_training else "valid"
        losses_names = self.losses_names + ["total"]
        total_loss = sum(self.running_losses)
        losses = self.running_losses + [total_loss]
        for loss, name in zip(losses, losses_names):
            loss /= self.running_losses_steps
            logger.info(f"{name}: {loss}")
            self.log_writer.add_scalar(f"{name}_{tag_suffix}", loss, global_step=epoch)

        self.last_total_loss = total_loss / self.running_losses_steps
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
        net_ins = self._get_filtered_input(self.dummy_input_train)
        net_outs = self.net(*net_ins)
        net_outs = [x.flatten().cpu() for x in net_outs]

        plt.figure(figsize=self.figsize)
        plt.violinplot(net_outs)
        plt.title("model outputs")
        plt.grid()
        plt.xlabel("output index")
        # plt.ylabel("")
        fig = plt.gcf()
        self.log_writer.add_figure("outputs", fig, epoch)
        plt.close()

    def _log_graph(self) -> None:
        """
        Saves the model graph.
        """
        x = self._get_filtered_input(self.dummy_input_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.log_writer.add_graph(self.net, x)
            except Exception:
                pass

    def _log_yaml(self) -> None:
        """
        Saves the yaml text on Tensorboard
        """
        # reading yaml
        with open(self.config_path) as f:
            txt = f.read()

        # adding also the parameters
        txt += f" parameters: {self._get_model_parameters() * 1e-3} K"

        self.log_writer.add_text("config.yaml", txt, 0)

    # = = = = = = = = = = = = = = = = = = = = = =
    #                Profiler
    # = = = = = = = = = = = = = = = = = = = = = =
    def _get_profiler(self) -> torch.profiler.profile:
        """
        Return the profiler context manager.

        Returns
        -------
        torch.profiler.profile
            Pytorch profiler
        """
        prof = (
            torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logs_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            if self.enable_profiling
            else nullcontext()
        )
        return prof

    def _start_profiling(self) -> None:
        """
        Starts the profiling.
        """
        self.profiler.__enter__()
        logger.info("profiler started")

    def _stop_profiling(self) -> None:
        """
        Stops the profiling.
        """
        self.profiler.__exit__(None, None, None)
        logger.info("profiler stopped")

        # leaving

    # = = = = = = = = = = = = = = = = = = = = = =
    #                Utilities
    # = = = = = = = = = = = = = = = = = = = = = =
    def _remove_extra_dim(self, data: List[Tensor]) -> List[Tensor]:
        """
        Removes extra starting dimensions from the
        data loaded.

        Parameters
        ----------
        data : List[Tensor]
            List of Tensors returned from the Dataloader

        Returns
        -------
        List[Tensor]
            data without a possible extra leading dimension
        """
        data = [x[0] if x.shape[0] == 1 else x for x in data]
        return data

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

    def _get_filtered_input(self, data: List[Tensor]) -> List[Tensor]:
        """
        Filters the Dataloader output based on net_ins_indices.

        Parameters
        ----------
        data : List[Tensor]
            Dataloader output

        Returns
        -------
        List[Tensor]
            Filtered inputs for the network
        """
        return [data[i] for i in self.net_ins_indices]

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
