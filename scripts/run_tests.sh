#!/bin/bash

test_dir="$(dirname "$0")/../tests"
test_dir="$(realpath "$test_dir")"
test_files=$(find "$test_dir" -name "test_*.py")

python "$test_dir/generate_test_data.py"
for t in $test_files; do
    echo "Running $t"
    python "$t" || exit 1
done
