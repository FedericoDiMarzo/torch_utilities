from pathimport import set_module_root
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
