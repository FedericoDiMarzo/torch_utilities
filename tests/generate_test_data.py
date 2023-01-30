from pathimport import set_module_root
from loguru import logger
from pathlib import Path
import numpy as np
import yaml
import h5py

set_module_root(".")
from torch_utilities import save_audio


def create_data_dir():
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)


def get_test_data_dir():
    path = Path(__file__).parent / "test_data"
    if not path.exists():
        raise FileNotFoundError("run generate_test_data.py first")
    return path


def generate_wavs(sample_rate: int = 16000, duration: float = 1):
    """
    Generate a stereo and a mono wav files
    of random noise.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate, by default 16000 Hz
    duration : float, optional
        Duration of the files, by default 1 s
    """
    data_dir = get_test_data_dir()
    files = [f"{x}.wav" for x in ("mono", "stereo")]
    channels = [1, 2]
    for n, c in zip(files, channels):
        x = np.stack(
            [np.random.uniform(-1, 1, duration * sample_rate) for _ in range(c)]
        )
        save_audio(data_dir / n, x, sample_rate)


def generate_hdf5(groups: int = 10, group_len: int = 16):
    data_layout = ["x", "y_true"]
    with h5py.File(get_test_data_dir() / "dataset.hdf5", "w") as ds:
        for i in range(groups):
            g = ds.create_group(f"group_{i}")
            for layout in data_layout:
                g_name = layout
                g.create_dataset(g_name, data=np.ones((group_len, 8)) * i)


def generate_yaml():
    """
    Generate a yaml dummy configuration.
    """
    data = {
        "section1": {
            "param1": "test",
            "param2": 42,
            "param3": ["a", "b", "c"],
        },
        "section2": {
            "param4": 12.43,
        },
    }

    with open(get_test_data_dir() / "test.yml", "w") as f:
        yaml.dump(data, f)


def generate_test_model_config(name: str, overfit_mode: bool):
    txt = f"""
---
general:
  test_param: 1234
  
training:
  learning_rate: 0.001
  weight_decay: 0.002
  overfit_mode: {overfit_mode}
  max_epochs: 1
  batch_size: 16
  log_every: 1
  num_workeres: 1
  losses_weights:
    - 1 
    - 2 
    - 3 
    - 4 
    """
    tag = "_overfit" if overfit_mode else ""
    model_dir = get_test_data_dir() / f"test_model{tag}"
    model_dir.mkdir(exist_ok=True)
    config_path = model_dir / "config.yml"
    with open(config_path, "wt") as f:
        f.write(txt)


def main():
    logger.info("generating test data")
    create_data_dir()
    generate_wavs()
    generate_hdf5()
    generate_yaml()
    generate_test_model_config("test_model", False)
    generate_test_model_config("test_model", True)
    logger.info("generation complete")


if __name__ == "__main__":
    main()
