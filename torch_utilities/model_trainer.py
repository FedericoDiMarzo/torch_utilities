from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch.utils.tensorboard import SummaryWriter
from contextlib import suppress, nullcontext
from torch.utils.data import DataLoader
from pathimport import set_module_root
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch import nn, Tensor
import torch_utilities as tu
from loguru import logger
from pathlib import Path
from abc import ABC
import numpy as np
import warnings
import torch
import math
import yaml
import abc

set_module_root(".")
from torch_utilities.common import DotDict
from torch_utilities.pytorch import load_checkpoints, sort_checkpoints

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
        save_buffer_maxlen: int = 10,
        enable_profiling: bool = False,
        reset_epoch: bool = False,
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
            by default 10
        enable_profiling : bool, optional
            If True runs one full run of the train dataset and
            logs to tensorboard the profiling information, by default False
        reset_epoch : bool, optional
            If set to True resets the epoch to 0 after loading the checkpoints,
            by default False


        config.yml [training] parameters
        ----------
        max_epochs : int, optional
            Max number of epochs, by default 100
        learning_rate : float, optional
            Optimizer learning rate starting value, by default 1e-5
        overfit_mode : bool, optional
            Enables overfit mode, by default False
        weight_decay : float, optional
            Optimizer weight_decay, by default 0
        losses_weights : List[float], optional
            Per-loss gains, by default all ones
        log_every : int, optional
            Frequency of the logs (over the dataset iterations),
            by default 100
        gradient_clip_value : float, optional
            Maximum gradient update value, by default no gradient clipping
        """
        self.device = tu.get_device()

        # explicit attributes
        self.model_path = model_path
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.losses = losses
        self.net_ins_indices = net_ins_indices or [0]
        self.losses_names = losses_names or self._default_losses_names()
        self.optimizer_class = optimizer_class
        self.enable_profiling = enable_profiling
        self.reset_epoch = reset_epoch

        # configuration attributes
        self.config_path = self.model_path / "config.yml"
        self.config = tu.Config(self.config_path)
        self.learning_rate = self._from_config("learning_rate", float, 1e-5)
        self.weight_decay = self._from_config("weight_decay", float, 0)
        self.log_every = self._from_config("log_every", int, 100)
        self.max_epochs = self._from_config("max_epochs", int, 100)
        self.overfit_mode = self._from_config("overfit_mode", bool, False)
        self.losses_weights = self._from_config(
            "losses_weights", np.array, np.ones(len(self.losses))
        )
        self.losses_weights = self.losses_weights.astype(float)

        gcv = self._from_config("gradient_clip_value", float, -1)
        self.gradient_clip_value = gcv if (gcv > 0) else None

        # other dirs
        self.checkpoints_dir = model_path / "checkpoints"
        self.checkpoint_monitoring_file = self.checkpoints_dir / "ckpt.yml"
        self.logs_dir = model_path / "logs"
        [d.mkdir(exist_ok=True) for d in (self.checkpoints_dir, self.logs_dir)]

        # model and running_losses setup
        self.start_epoch = 0
        self.save_buffer = []
        self.save_buffer_maxlen = save_buffer_maxlen
        self.net = self._load_model()
        self.optim_state = None
        self.optimizer = self._setup_optimizer()
        self.disable_optimization = False
        self.running_losses = None
        self.running_losses_steps = 0
        self.total_loss_name = "total"
        self._reset_running_losses()

        # extra stuff
        self.is_validation = False
        self.log_writer = SummaryWriter(self.logs_dir)
        self.figsize = (8, 6)
        self.dummy_input_train = self._get_dummy_data(True)
        self.dummy_input_valid = self._get_dummy_data(False)
        self.last_total_loss = 1e10
        self.last_computed_metric = -1e10  # used to select the best checkpoints
        self.profiler = self._get_profiler()  # null context manager if enable_profiling==False

    # = = = = = = = = = = = = = = = = = = = = = =
    #         Public getters/setters
    # = = = = = = = = = = = = = = = = = = = = = =
    def get_model(self) -> nn.Module:
        """
        Gets the model instance.

        Returns
        -------
        nn.Module
            Model instance
        """
        return self.net

    def is_training(self) -> bool:
        """
        Returns True if the train steps are running.

        Returns
        -------
        bool
            True if the train steps are running.
        """
        return not self.is_validation

    def get_losses(self) -> List[Callable]:
        """
        Gets the loss functions.

        Returns
        -------
        List[Callable]
            List of loss functions in the same order as
            they're passed to the constructor
        """
        return self.losses

    def get_losses_weights(self) -> List[float]:
        """
        Gets the weight of losses.

        Returns
        -------
        List[float]
            List of loss weights in the same order as
            they're passed to the constructor
        """
        return self.losses_weights

    def get_learning_rate(self) -> float:
        """
        Gets the current learning rate.

        Returns
        -------
        float
            Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def set_learning_rate(self, value: float) -> None:
        """
        Setter for learning_rate.

        Parameters
        ----------
        value : float
            New learning rate
        """
        for g in self.optimizer.param_groups:
            g["lr"] = value

    def set_weight_decay(self, value: float) -> None:
        """
        Setter for weight_decay.

        Parameters
        ----------
        value : float
            New learning rate
        """
        for g in self.optimizer.param_groups:
            g["weight_decay"] = value

    def get_tensorboard_writer(self) -> SummaryWriter:
        """
        Getter to the tensorboard log writer

        Returns
        -------
        SummaryWriter
            tensorboard log writer
        """
        return self.log_writer

    # = = = = = = = = = = = = = = = = = = = = = =
    #             Training loop
    # = = = = = = = = = = = = = = = = = = = = = =
    def start_training(self) -> None:
        """
        Trains a model.
        """
        # logging
        logger.info(f"device selected: {self.device}")
        logger.info(f"model: {self.model_path.name}")
        logger.info(f"parameters: {self._get_model_parameters() / 1e3} K")
        if self.overfit_mode:
            logger.info("overfit mode on")
        if self.enable_profiling:
            logger.info("profiler on: stopping after one epoch")
        else:
            logger.info("learning rate scheduler enabled")
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
            self.is_validation = False
            for i, data in enumerate(self.train_dl):
                data = self._remove_extra_dim(data)
                # in overfit mode dummy data is used instead of real data
                data = self.dummy_input_train if self.overfit_mode else data
                self._train_step(data, epoch)
                if i % self.log_every == 0 and i != 0:
                    logger.info(f"batch [{i}/{len(self.train_dl)}]")
                    self._log_losses(epoch=epoch)
                #
            self._log_gradients(epoch)
            self._log_losses(epoch=epoch)

            _log_data = lambda t: self.dummy_input_train if t else self.dummy_input_valid
            with torch.no_grad():
                logger.info("logging tensorboard train data")
                self.tensorboard_logs(_log_data(True), epoch=epoch)
                self._log_outs(epoch)

                # leaving the training  ~ ~
                self._stop_profiling()  # <- - profiler off
                if self.enable_profiling:
                    logger.info(f"{self.model_path.name} profilation complete")
                    return
                # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

                if not self.overfit_mode:
                    # validation
                    self.net.eval()
                    self.is_validation = True
                    for i, data in enumerate(self.valid_dl):
                        data = self._remove_extra_dim(data)
                        self._valid_step(data, epoch)
                    self._log_losses(epoch=epoch)
                    logger.info("logging tensorboard valid data")
                    self.tensorboard_logs(_log_data(False), epoch=epoch)

            metric_log_data = _log_data(self.is_training())
            self.last_computed_metric = self.apply_metric(metric_log_data)
            self._save_model(epoch)
            self.on_epoch_end(epoch)
            self._check_nan()

        logger.info(f"{self.model_path.name} training complete")  # - = - § >>
        self.on_train_end()

    def _train_step(self, data: List[Tensor], epoch: int) -> None:
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

        # enable custom optimization schemes ~ ~ ~ ~ ~ ~
        if not self.disable_optimization:
            total_loss = sum(_losses)
            total_loss.backward()
            # gradient clipping
            if self.gradient_clip_value is not None:
                nn.utils.clip_grad_value_(
                    self.net.parameters(),
                    clip_value=self.gradient_clip_value,
                )
            self.optimizer.step()
            # for logging
            self._update_running_losses(_losses)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        self.on_train_step_end(epoch)

        # profiler update
        if self.enable_profiling:
            self.profiler.step()

    def _valid_step(self, data: List[Tensor], epoch) -> None:
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

    def apply_metric(self, data: List[Tensor]) -> float:
        """
        Defines the metric used to keep the best checkpoints.
        The higher the better.

        Parameters
        ----------
        data : List[Tensor]
            List of Tensors loaded by the dataloader

        Returns
        -------
        float
            Value of the metric, the higher the better
        """
        # by default the metric is the inverse of the last_total_loss
        return -float(self.last_total_loss)

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
    #         Model loading/saving
    # = = = = = = = = = = = = = = = = = = = = = =
    def _load_model(self) -> nn.Module:
        """
        Loads a model from its class and configuration file.

        Returns
        -------
        nn.Module
            Loaded model
        """
        m = self.model(self.config_path)
        m.to(self.device)

        # load checkpoint if it exists
        if self._prev_train_exists():
            epoch, model_state, optim_state = self._load_checkpoint()
            self.start_epoch = 0 if self.reset_epoch else epoch
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

    def _load_checkpoint(self) -> Tuple[int, Dict, Dict, Dict]:
        """
        Loads the best checkpoint.

        Returns
        -------
        Tuple[int, Dict, Dict]
            (epoch, model_state, optim_state)
        """
        # choosing the best checkpoint
        self._load_checkpoint_monitoring()
        best_checkpoint = self.save_buffer[0][0]
        best_checkpoint = self.checkpoints_dir / best_checkpoint

        # loading and parsing
        best_checkpoint = DotDict(torch.load(best_checkpoint, map_location=self.device))
        epoch = best_checkpoint.epoch + 1
        model_state = best_checkpoint.model_state
        optim_state = best_checkpoint.optim_state
        return epoch, model_state, optim_state

    def _save_model(self, epoch: int) -> None:
        """
        Saves the model in the checkpoints folder.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        # saving
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{epoch+1}.ckpt"
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
        Pushes a new checkpoint into the save_buffer and keeps track
        of the best ones.

        Parameters
        ----------
        checkpoint_name : str
            Name of the checkpoint
        """
        # sorting by the highest metric
        self.save_buffer.append((checkpoint_name, float(self.last_computed_metric)))
        self._sort_checkpoints()

        self._delete_worse_checkpoints()
        self._save_checkpoint_monitoring()

    def _save_checkpoint_monitoring(self) -> None:
        """
        Saves the state of the save_buffer.
        """
        with open(self.checkpoint_monitoring_file, "w") as f:
            yaml.dump(self.save_buffer, f)

    def _load_checkpoint_monitoring(self) -> None:
        """
        Loads the checkpoints.
        """
        self.save_buffer = load_checkpoints(self.checkpoint_monitoring_file)

    def _sort_checkpoints(self) -> None:
        """
        Sorts the save_buffer based on the metric value.
        """
        self.save_buffer = sort_checkpoints(self.save_buffer)
        self.save_buffer = self.save_buffer[: self.save_buffer_maxlen]

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
    @abc.abstractclassmethod
    def tensorboard_logs(self, raw_data: List[Tensor], epoch: int) -> None:
        """
        Additional tensorboard logging.

        Parameters
        ----------
        raw_data : List[Tensor]
            Dataset input
        epoch : int
            Current epoch
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
        ds = self.train_dl if is_training else self.valid_dl
        data = [x.to(self.device) for x in ds.dataset[[0, 1]]]
        return data

    def _log_losses(self, epoch: int) -> None:
        """
        Logs running_losses.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        if self.running_losses_steps == 0:
            # nothing to log
            return

        tag_suffix = "train" if self.is_training() else "valid"
        losses_names = self.losses_names + [self.total_loss_name]
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
        plt.title("model gradients")
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
        if self.enable_profiling:
            self.profiler.__enter__()
            logger.info("profiler started")

    def _stop_profiling(self) -> None:
        """
        Stops the profiling.
        """
        if self.enable_profiling:
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

    def _check_nan(self) -> None:
        """
        Checks if the last loss is NaN.

        Raises
        ----------
        RuntimeError
            In case a NaN value is encountered
        """
        if math.isnan(self.last_total_loss):
            err_msg = "NaN value encountered in last_total_loss"
            logger.error(err_msg)
            raise RuntimeError(err_msg)
