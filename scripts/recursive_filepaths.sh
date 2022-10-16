#!/bin/bash

function usage {
    echo "usage: $0 root_dir [-h |-f extension]"
    echo "description: returns a the fullpath of all the files starting from a root_dir"
    echo "-h         show help"
    echo "root_dir   starting directory"
    echo "extension  filter for file type"
    exit 0
}

# arg parsing
while getopts ":hf:" o; do
    case "${o}" in
        f)
            ext=${OPTARG} ;;
        *)
            usage ;;
    esac
    shift "$((OPTIND - 1))"
done

[ $# -eq 0 ] && usage
root=$(realpath $1)

# pipeline
find "$root" | grep "\.$ext\b"
