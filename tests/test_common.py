from typing import Tuple
from pathimport import set_module_root
from ray import tune
from torch import nn
import numpy as np
import itertools
import unittest
import torch

set_module_root("../torch_utils")
import torch_utilities as tu
from tests.generate_test_data import get_test_data_dir


def _setup() -> None:
    torch.manual_seed(984)
    np.random.seed(901)
    tu.set_device("auto")
    torch.set_grad_enabled(False)


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.config_path = get_test_data_dir() / "test.yml"
        self.config = tu.Config(self.config_path)

    def test_get_str(self):
        y_true = "test"
        y_test = self.config.get("section1", "param1", str)
        self.assertEqual(y_true, y_test)

    def test_get_int(self):
        y_true = 42
        y_test = self.config.get("section1", "param2", int)
        self.assertEqual(y_true, y_test)

    def test_get_list(self):
        y_true = ["a", "b", "c"]
        y_test = self.config.get("section1", "param3", list)
        self.assertEqual(y_true, y_test)

    def test_get_float(self):
        y_true = 12.43
        y_test = self.config.get("section2", "param4", float)
        self.assertAlmostEqual(y_true, y_test)

    def test_default_none(self):
        y_true = None
        y_test = self.config.get("section1", "param10")
        self.assertEqual(y_true, y_test)

    def test_default(self):
        y_true = "default"
        y_test = self.config.get("section1", "param10", default="default", _type=int)
        self.assertEqual(y_true, y_test)

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

    def test_get_ray_tune_params(self):
        params = self.config.get_ray_tune_params()
        sampling_methods = set(params.keys())
        params_names = ["weight_decay", "loss_weight_0", "depth", "learning_rate", "list_choice"]
        self.assertEqual(sampling_methods, set(params_names))
        self.assertEqual(type(params), dict)
        params = tu.DotDict(params)

        # random variable type
        self.assertEqual(type(params.weight_decay), tune.search.sample.Float)
        self.assertEqual(type(params.learning_rate), tune.search.sample.Float)
        self.assertEqual(type(params.loss_weight_0), tune.search.sample.Float)
        self.assertEqual(type(params.depth), tune.search.sample.Categorical)
        self.assertEqual(type(params.list_choice), tune.search.sample.Categorical)

        # type after sampling
        self.assertEqual(type(params.weight_decay.sample()), float)
        self.assertEqual(type(params.learning_rate.sample()), float)
        self.assertEqual(type(params.loss_weight_0.sample()), float)
        self.assertEqual(type(params.depth.sample()), int)
        self.assertEqual(type(params.list_choice.sample()), list)


class TestGeneric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        pass

    def test_pack_many(self):
        xs = [1, 2, 3]
        ys = [4, 5, 6]
        zss = tu.pack_many(xs, ys)
        self.assertEqual(zss, [(1, 4), (2, 5), (3, 6)])

    def test_split_complex(self):
        mod = (np, torch)
        chl = (1, 2, 4, 8)
        params = itertools.product(mod, chl)
        xs = [m.ones((1, c, 16), dtype=m.complex64) for m, c in params]
        for x, p in zip(xs, params):
            with self.subTest(p=p):
                y = tu.split_complex(x)
                c = y.shape[1]
                _ones = mod.ones_like(y[:, : c // 2])
                _zeros = mod.zeros_like(y[:, : c // 2])
                self.assertEqual(c, x.shape[1] * 2)
                self.assertTrue(mod.allclose(y[:, : c // 2]), _ones)
                self.assertTrue(mod.allclose(y[:, c // 2 :]), _zeros)

    def test_pack_complex(self):
        mod = (np, torch)
        chl = (2, 4, 8)
        params = itertools.product(mod, chl)
        xs = [m.ones((1, c, 16)) for m, c in params]
        for x in xs:
            with self.subTest(x=x):
                y = tu.pack_complex(x)
                self.assertEqual(y.shape[1], x.shape[1] // 2)

    def test_one_hot_quantization(self):
        min_max = ((-1, 1), (0, 65535), (11.1, 23.9))
        steps = (8, 256, 1024)
        params = itertools.product(min_max, steps)
        for p in params:
            (min_, max_), s = p
            with self.subTest(p=p):
                C, T, F = 2, 10, 32
                x = torch.rand(1, C, T, F) * (max_ - min_) + min_
                y = tu.one_hot_quantization(x, s, min_, max_)
                self.assertEqual(y.shape, (1, s, C, T, F))
                self.assertEqual(y.sum().item(), C * T * F)

    def test_invert_one_hot(self):
        n_dims = (2, 3, 4)
        steps = (3, 10, 256)
        params = itertools.product(n_dims, steps)
        for d, s in params:
            with self.subTest(d=d, s=s):
                x = torch.rand([3] * d) * s
                x = torch.floor(x).clip(0, s - 1)
                y = tu.one_hot_quantization(x, s, 0, s)
                x_hat = tu.invert_one_hot(y)
                self.assertTrue(x.equal(x_hat))

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


class TestCosineScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        self.values = [(0, 1), (100, -100)]
        self.total_epochs = (10, 21)
        self.iterations_per_epochs = (1, 7)
        self.warmup_epochs = (0, 3)
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
                # end
                print(b)
                print(scheduling[-1])
                exit(-1)
                self.assertLess(np.abs(scheduling[-1] - b), 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
