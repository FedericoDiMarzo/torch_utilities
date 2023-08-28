# Torch Utilities
**Torch Utilities**  is a Python module that provides various tools related to audio and deep learning using the PyTorch framework. The module simplifies the process of building and training deep learning models for audio data, making it easier to experiment and develop custom models and pipelines. With its *user-friendly API*, the module also enables researchers and engineers to quickly prototype new ideas and evaluate their performance. Whether you're an experienced deep learning practitioner or just starting out, **Torch Utilities** has the tools you need to get the job done efficiently.

## Disclaimer
**Torch Utilities** is developed as a personal set of tools and is provided as-it-is, without any warranties or guarantees. The API and utilities may change in the future as the module continues to evolve. While effort is made to maintain compatibility with previous versions, users are advised to carefully consider the stability of the API before using **Torch Utilities** in production environments.

## Installation
You can install **Torch Utilities** using pip.
```bash
pip install torch_utilities
```

## Running The Tests
To verify the correctness of the code and run the tests, you can execute the following script.
```bash
chmod +x scripts/*.sh
./scripts/run_tests.sh
```
This will run a suite of tests to ensure that the module is functioning as expected. If any tests fail, it may indicate that there is a bug in the code or that some aspect of the API has changed. In such cases, we encourage you to open an issue on the repository so that we can help resolve the problem.


## Module Documentation
To read the API documentation of the module you need to clone the repository locally and run pdoc.
```bash
git clone git@github.com:FedericoDiMarzo/torch_utilities.git
pdoc --docformat numpy torch_utilities/torch_utilities
```
The documentation will then be accessible at the address http://localhost:8080 .

## Audio Tools
**Torch Utilities**  includes a comprehensive set of tools for handling audio signals, making it easy for you to preprocess, augment, and manipulate audio data. With simplified *IO utilities*, you can quickly and easily load audio data from disk and save the processed results, without worrying about low level details such as handling PyTorch devices. The module also includes functions for processing and extracting features from audio signals, allowing you to quickly transform raw audio into a format suitable for use with deep learning models.
These tools support both numpy ndarrays and PyTorch tensors, making it effortless to switch between the two and use the best tool for the job.

## Custom Audio Modules
**Torch Utilities** also features a collection of *custom layers* designed specifically for the *causal implementation* of deep neural networks. These layers are implemented for use in real-time audio processing applications, where maintaining causality is crucial. With these layers, you can quickly build and train models that can process audio data in a causal manner.

Moreover, **Torch Utilities** includes implementations of popular layers and architectures from the literature, giving you access to a wealth of proven and well-established building blocks. With these implementations, you can save time and effort in building and training your own models, as well as benefit from the extensive research and experimentation that has gone into these layers. Whether you're looking to implement state-of-the-art deep learning models for audio processing or just need some basic building blocks to get started, **Torch Utilities** has you covered.

## Flexible Data Loading And Training
 **Torch Utilities** also includes a set of tools to simplify model training and data loading. With these tools, you can spend less time writing boilerplate code and more time focusing on the specifics of your model and data.

The *data loading tools* allow you to create HDF5 datasets from a list of files using the command line, as well as providing Dataset classes for offline and online data loading. These classes make it easy to load, and augment your data, and can be customized to meet your specific needs. Whether you're working with large datasets or just need to quickly prototype a new idea, these tools make data loading a breeze.

The *model training tools*, on the other hand, allow you to easily provide flexible training scripts and perform model diagnostics. With these tools, you can avoid writing repetitive code and instead focus on the specifics of your model and data. The tools provide a convenient way to monitor and visualize your training progress, and they allow you to easily save and load your models and training results. 
