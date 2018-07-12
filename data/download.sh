#!/usr/bin/bash

# Make sure we are in the download directory.
cd "$(dirname "$0")"

# Remove training, validation and test directories if they exist and the
# parmater objects W and b.
rm -rf train/
rm -rf val/
rm -rf test/
rm -f W.npy
rm -f b.npy

# Make necessary directories.
mkdir train
mkdir val
mkdir test

# Move into Python virtual environment (unnecessary if not using it).
source ../env/bin/activate
#echo $VIRTUAL_ENV

# Generate the data with the Python script.
python3 _generate_data.py
