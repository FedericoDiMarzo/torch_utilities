from contextlib import nullcontext
from typing import Callable, List, Tuple, Type, Optional
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Rprop, Optimizer
from pathimport import set_module_root
from itertools import product
from torch import Tensor, nn
from loguru import logger
from pathlib import Path
import numpy as np
import unittest
import torch

set_module_root("../torch_utils")
from tests.generate_test_data import get_test_data_dir
import torch_utilities as tu

# = = SETUP = = = = = = = = = = = = = = = = = = = = = = =


def _setup() -> None:
    np.random.seed(901)
    tu.set_device("auto")
    torch.manual_seed(984)
    torch.set_grad_enabled(True)
    logger.disable("torch_utils.model_trainer")


# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class ModelDummy(nn.Module):
    def __init__(self, config_path: Path) -> None:
        """
        Test model.
        """
        super().__init__()
        self.config = tu.Config(config_path)
        self.test_param = self.config.get("general", "test_param", int)
        self.layer = nn.Identity()
        self.batchnorm = nn.BatchNorm1d(8)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.layer(x)
        x = self.batchnorm(x)
        return [x]


# = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class ModelTrainerDummy(tu.ModelTrainer):
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
        enable_profiling: bool = False,
    ) -> None:
        """
        ModelTrainer implementation.
        """
        super().__init__(
            model_path,
            model,
            train_dl,
            valid_dl,
            optimizer_class,
            losses,
            net_ins_indices,
            losses_names,
            enable_profiling=enable_profiling,
        )
        self.net = self.net.to(float)

        # callbacks flags
        self.on_train_begin_flag = False
        self.on_train_end_flag = False
        self.on_train_step_begin_flag = False
        self.on_train_step_end_flag = False
        self.on_valid_step_begin_flag = False
        self.on_valid_step_end_flag = False

    def apply_losses(self, data: List[Tensor], net_outs: List[Tensor]) -> List[Tensor]:
        y_hat = net_outs[0]
        y_true = data[1]
        loss_values = [loss(y_hat, y_true) for loss in self.losses]
        return loss_values

    def tensorboard_logs(self, raw_data: List[Tensor], epoch: int, is_training: bool) -> None:
        pass

    def on_train_begin(self) -> None:
        self.on_train_begin_flag = True

    def on_train_end(self) -> None:
        self.on_train_end_flag = True

    def on_train_step_begin(self, epoch) -> None:
        self.on_train_step_begin_flag = True

    def on_train_step_end(self, epoch) -> None:
        self.on_train_step_end_flag = True

    def on_valid_step_begin(self, epoch) -> None:
        self.on_valid_step_begin_flag = True

    def on_valid_step_end(self, epoch) -> None:
        self.on_valid_step_end_flag = True


# = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class TestModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.model_path = (get_test_data_dir() / n for n in ("test_model", "test_model_overfit"))
        self.loader_type = (tu.HDF5Dataset, tu.HDF5OnlineDataset)
        self.optimizer = (Adam, Rprop)  # with and without weight decay setting
        self.n_losses = (4,)
        self.params = product(
            self.model_path,
            self.loader_type,
            self.optimizer,
            self.n_losses,
        )
        self.params = list(self.params)
        self.batch_size = 16
        self.clear_model_folders()

    def clear_model_folders(self):
        """
        Cleans the model folders.
        """
        for model in self.model_path:
            not_config = list(set(model.glob("*")) - set(model.glob("*.yml")))
            [f.unlink for f in not_config]

    def get_model_trainer(self, params: Tuple, enable_profiling: bool = False) -> ModelTrainerDummy:
        """
        Initializes the model trainer.

        Parameters
        ----------
        params : Tuple
            Parameters
        enable_profiling : bool, optional
            If True enables the profiler, by default False

        Returns
        -------
        ModelTrainerDummy
            ModelTrainer implementation
        """
        (
            model_path,
            loader_type,
            optimizer,
            n_losses,
        ) = params
        dataloader = self.get_dataloader(loader_type)
        trainer = ModelTrainerDummy(
            model_path=model_path,
            model=ModelDummy,
            train_dl=dataloader,
            valid_dl=dataloader,
            optimizer_class=optimizer,
            losses=[nn.MSELoss()] * n_losses,
            net_ins_indices=[0],
            losses_names=[f"test_loss_{i}" for i in range(n_losses)],
            enable_profiling=enable_profiling,
        )
        return trainer

    def get_dataloader(self, loader_type: Type[Dataset]) -> DataLoader:
        """
        Initialize a specific DataLoader.

        Parameters
        ----------
        loader_type : Type[Dataset]
            Dataset type

        Returns
        -------
        DataLoader
            DataLoader instance
        """
        dataset_path = get_test_data_dir() / "dataset.hdf5"
        data_layout = ["x", "y_true"]

        if loader_type == tu.HDF5Dataset:
            loader = tu.get_hdf5_dataloader(
                dataset_path=dataset_path,
                data_layout=data_layout,
                batch_size=self.batch_size,
            )
        else:
            n = 2
            ds = tu.HDF5OnlineDataset(
                dataset_paths=[dataset_path] * n,
                data_layouts=[data_layout] * n,
                batch_size=self.batch_size,
                total_items=self.batch_size,
            )
            loader = DataLoader(ds)

        return loader

    def test_init(self):
        for p in self.params:
            with self.subTest(p=p):
                self.get_model_trainer(p)

    def test_model_config(self):
        p = self.params[0]
        trainer = self.get_model_trainer(p)
        self.assertEqual(trainer.net.test_param, 1234)

    def test_learning_rate(self):
        p = self.params[0]
        trainer = self.get_model_trainer(p)
        optim = trainer.optimizer
        self.assertAlmostEqual(optim.param_groups[-1]["lr"], 0.001)

    def test_weight_decay(self):
        p = self.params[0]
        trainer = self.get_model_trainer(p)
        optim = trainer.optimizer
        self.assertAlmostEqual(optim.param_groups[-1]["weight_decay"], 0.002)

    def test_log_every(self):
        p = self.params[0]
        trainer = self.get_model_trainer(p)
        self.assertAlmostEqual(trainer.log_every, 1)

    def test_losses_weights(self):
        for p in self.params:
            with self.subTest(p=p):
                trainer = self.get_model_trainer(p)
                losses_w = trainer.losses_weights.tolist()
                target = list(range(1, len(losses_w) + 1))
                self.assertListEqual(losses_w, target)

    def test_callbacks(self):
        params = self.params[0], self.params[-1]
        for p in params:
            with self.subTest(p=p):
                trainer = self.get_model_trainer(p)
                trainer.start_training()
                overfit = "overfit" in str(p[0])
                self.assertEqual(
                    [
                        trainer.on_train_begin_flag,
                        trainer.on_train_end_flag,
                        trainer.on_train_step_begin_flag,
                        trainer.on_train_step_end_flag,
                        trainer.on_valid_step_begin_flag,
                        trainer.on_valid_step_end_flag,
                    ],
                    [True] * 4 + [False] * 2 if overfit else [True] * 6,
                )

    def test_profiler(self):
        p = self.params[0]
        for is_profiling in (False, True):
            with self.subTest(is_profiling=is_profiling):
                trainer = self.get_model_trainer(p, enable_profiling=is_profiling)
                expected = torch.profiler.profile if is_profiling else nullcontext
                self.assertEqual(type(trainer.profiler), expected)

    def test_get_dummy_data(self):
        params = self.params
        is_training = (False, True)
        params = product(params, is_training)
        for p, it in params:
            is_hdf5_dataset = p[1] == tu.HDF5Dataset
            with self.subTest(p=p, it=it):
                trainer = self.get_model_trainer(p)
                xs = trainer._get_dummy_data(it)
                y = 2 if is_hdf5_dataset else 4
                self.assertEqual(len(xs), y)
                y = 2 if is_hdf5_dataset else self.batch_size
                [self.assertEqual(x.shape[0], y) for x in xs]
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
