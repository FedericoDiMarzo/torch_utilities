#!/bin/bash

version=$(cat setup.cfg | grep version | cut -d " " -f3)

echo "Building torch_utilities version $version"

rm -rf dist
python -m build
twine upload dist/*
