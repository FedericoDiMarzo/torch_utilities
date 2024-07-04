# Torch Utilities

**Torch Utilities** is a collection of tools and utilities for working with PyTorch in the audio domain.

## Disclaimer

**Torch Utilities** is developed as a personal set of tools and is provided as-it-is, without any warranties or guarantees. The API and utilities may change in the future as the module continues to evolve. While effort is made to maintain compatibility with previous versions, users are advised to carefully consider the stability of the API before using **Torch Utilities** in production environments.

## Installation

You can install **Torch Utilities** using pip.

```bash
pip install torch_utilities
```

## Running The Tests

To run the tests you need to clone the repository locally and run use `pytest`.

```bash
git clone git@github.com:FedericoDiMarzo/torch_utilities.git
pip install -e torch_utilities[dev]
pytest torch_utilities/tests
```

If any tests fail, it may indicate that there is a bug in the code or that some aspect of the API has changed. In such cases, we encourage you to open an issue on the repository so that we can help resolve the problem.

## Module Documentation

To read the API documentation of the module after cloning the repository, you can use `pdoc` to generate the documentation and serve it locally.

```bash
pdoc --docformat numpy torch_utilities/torch_utilities
```

The documentation will then be accessible at the address http://localhost:8080 .

## How to explore the tools

The most relevant function and classes are available in the main namespace of the module. You can import them directly and use them in your code.

The source code is divided into

- `audio`: Utilities working mostly on waveforms
- `augmentation`: Data augmentation for audio signals
- `io`: Input/output utilities
- `metrics`: Various metrics for audio signals
- `modules`: **PyTorch modules for audio processing**
- `utilities`: General utilities.

