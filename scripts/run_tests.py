from pathimport import set_module_root
from argparse import ArgumentParser
from typing import Dict
import subprocess
import sys

set_module_root("../tests", )
sys.path = [sys.path[-1]] + sys.path
from tests import generate_test_data, test_dir


def main():
    generate_test_data()
    subprocess.call([sys.executable, "-m", "unittest", "discover"])


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
