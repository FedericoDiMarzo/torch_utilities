from typing import Iterable, Tuple
from torch import Tensor, nn
import numpy as np
import itertools
import unittest
import torch

import torch_utilities as tu
from tests.generate_test_data import get_test_data_dir


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestPyTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        pass

    @torch.set_grad_enabled(True)
    def test_compute_gradients(self):
        keep_graph = (False, True)
        for k in keep_graph:
            x = torch.ones((3, 2), requires_grad=True)
            y = x**2
            grad = tu.compute_gradients(x, y, keep_graph=k)
            z = grad.sum()
            self.assertLess(z.item() - 2 * 6, 1e-8)
            if k:
                z.backward()
            else:
                with self.assertRaises(RuntimeError):
                    z.backward()

    def test_get_submodules(self):
        model = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Tanh(),
            nn.Sequential(
                nn.SELU(),
                nn.Sigmoid(),
            ),
        )
        modules = tu.get_submodules(model)
        modules_types = [type(m) for m in modules]
        expected = [nn.Identity, nn.ReLU, nn.Tanh, nn.SELU, nn.Sigmoid]
        self.assertListEqual(modules_types, expected)

    def test_load_checkpoints(self):
        checkpoint_file = get_test_data_dir() / "dummy_ckpt.yml"
        checkpoints = tu.load_checkpoints(checkpoint_file)
        self.assertEqual(len(checkpoints), 10)
        [self.assertEqual(x[0], f"checkpoint_{9-i}") for i, x in enumerate(checkpoints)]
        [
            self.assertLess(np.abs(x[1] - 9 + i), 1e-12)
            for i, x in enumerate(checkpoints)
        ]

    def test_sort_checkpoints(self):
        names = ("a", "b")
        scores = (-10.2, 11)
        checkpoints = list(zip(names, scores))
        checkpoints = tu.sort_checkpoints(checkpoints)
        self.assertEqual(checkpoints[0][0], "b")
        self.assertEqual(checkpoints[1][0], "a")

    @torch.set_grad_enabled(True)
    def test_freeze_model(self):
        module = nn.Conv2d(2, 2, 1)
        self.assertTrue(all(p.requires_grad for p in module.parameters()))
        tu.freeze_model(module)
        self.assertFalse(all(p.requires_grad for p in module.parameters()))


class TestCosineScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        # self.values = [(0, 1), (100, -100)]
        self.values = [
            (0, 1),
        ]
        # self.total_epochs = (10, 21)
        self.total_epochs = (10,)
        # self.iterations_per_epochs = (1, 7)
        self.iterations_per_epochs = (1,)
        # self.warmup_epochs = (0, 3)
        self.warmup_epochs = (0,)
        self.params = itertools.product(
            self.values,
            self.total_epochs,
            self.iterations_per_epochs,
            self.warmup_epochs,
        )

    def get_instance(self, params: Tuple) -> tu.CosineScheduler:
        vs, e, it, we = params
        scheduler = tu.CosineScheduler(*vs, e, it, we)
        return scheduler

    def test_compute_schedule(self) -> None:
        for p in self.params:
            vs, e, it, we = p
            a, b = vs
            with self.subTest(p=p):
                scheduler = self.get_instance(p)
                scheduling = scheduler.schedule
                warmup_steps = we * it
                # warmup
                warmup_start = a if (warmup_steps == 0) else (a / warmup_steps)
                self.assertLess(np.abs(warmup_start - scheduling[0]), 1e-12)
                # start
                self.assertLess(np.abs(scheduling[warmup_steps] - a), 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
