from pathimport import set_module_root
from pathlib import Path

set_module_root(".")

from tests.generate_test_data import main as generate_test_data

test_dir = Path(__file__).parent.absolute()
