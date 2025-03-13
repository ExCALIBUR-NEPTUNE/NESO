#!/bin/bash

# Sets up a Python virtual environment with NekPy and h5py, suitable for running regression test data generator.
#   Either
#     i) run with no args to try and autodetect NekPy's setup.py in ./nektar OR
#    ii) specify the path to a build directory containing setup.py as an argument

# Process args
setup_fname="setup.py"
if [ $# -eq 0 ]; then
    nektar_dir="$(find -L ../../nektar/builds -name "$setup_fname" -printf "%TY-%Tm-%Td %TT %p\n" | sort -n|tail -1|cut -d " " -f 3 | xargs -r dirname)"
    # Bail out if NekPy's setup file wasn't found
    if [ -z "$nektar_dir" ]; then
        echo "Failed to auto-detect Nektar build with Python support"
        echo " (Nektar must have been configured with BUILD_PYTHON=True and installed to generate $setup_fname)"
        exit 1
    fi
elif [ $# -eq 1 ]; then
    nektar_dir="$1"
else
    echo "Usage: $0 <path_to_nektar_build_dir> "
    exit 2
fi

# Check NekPy's setup file exists 
setup_path="$nektar_dir/$setup_fname"
if [ ! -f "$setup_path" ]; then
    echo "$setup_fname not found in $setup_path"
    exit 3
else 
    echo "Found $setup_path"
fi

# Remove existing environment if there is one
env_dir=".venv"
if [ -d "$env_dir" ]; then
    \rm -rf "$env_dir"; 
fi

# Generate and activate the venv
python3 -m venv "$env_dir"
. "$env_dir/bin/activate"
pip install wheel

# Install requirements, including NekPy
pip install -r "requirements.txt"
pip install "$nektar_dir"

printf "Activate the environment with\n   . $env_dir/bin/activate\n"