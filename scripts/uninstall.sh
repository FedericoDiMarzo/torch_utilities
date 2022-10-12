#!/bin/bash

# uninstall the scripts in the system
script_dir=$(realpath $(dirname -- "$0"))
script_path="/bin/recursive_filepaths"  
sudo rm "$script_path" || exit -1
echo "recursive_filepaths script uninstalled"