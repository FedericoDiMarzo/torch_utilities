from pathimport import set_module_root
from torch import Tensor
import numpy as np
import itertools
import unittest
import torch

set_module_root("../torch_utils", prefix=True)
import torch_utils as tu
from tests.generate_test_data import get_test_data_dir
from torch_utils import repeat_test, set_device, get_device

torch.manual_seed(984)
np.random.seed(901)
set_device("auto")


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

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


class TestGeneric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

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
        for x in xs:
            with self.subTest(x=x):
                y = tu.split_complex(x)
                c = y.shape[1]
                self.assertEqual(c, x.shape[1] * 2)
                self.assertTrue(np.allclose(y[:, : c // 2]), ) # TODO : TeSt
                self.assertTrue(np.allclose(y[:, c // 2 :]), ) # TODO : TeSt

    def test_pack_complex(self):
        mod = (np, torch)
        chl = (2, 4, 8)
        params = itertools.product(mod, chl)
        xs = [m.ones((1, c, 16)) for m, c in params]
        for x in xs:
            with self.subTest(x=x):
                y = tu.pack_complex(x)
                self.assertEqual(y.shape[1], x.shape[1] // 2)


if __name__ == "__main__":
    unittest.main()
