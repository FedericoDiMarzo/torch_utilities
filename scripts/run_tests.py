from pathimport import set_module_root
from argparse import ArgumentParser
from loguru import logger
from typing import Dict
import unittest

set_module_root("../tests")
from tests import generate_test_data, test_dir


def main():
    generate_test_data()
    logger.info(f"running tests from {test_dir}")
    test_loader = unittest.TestLoader()
    tests = test_loader.discover(test_dir, "test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(tests)


def parse_args() -> Dict:
    """
    Parses command line arguments.

    Returns
    -------
    Dict
        Parsed arguments
    """
    desc = "Runs all the test relative to the module torch_utils."
    argparser = ArgumentParser(description=desc)
    return argparser.parse_args()


if __name__ == "__main__":
    main()
