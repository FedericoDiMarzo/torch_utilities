#!/bin/bash

# install the scripts in the system
script_dir=$(realpath $(dirname -- "$0"))
script_path="$script_dir/recursive_filepaths.sh"  
chmod +x "$script_path"
sudo cp "$script_path" "/bin/recursive_filepaths" || exit -1
echo "recursive_filepaths script installed"