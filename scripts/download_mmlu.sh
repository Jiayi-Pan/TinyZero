#!/bin/bash

# This script downloads and extracts the MMLU dataset.

if [ -d "data/data" ]; then
    echo "MMLU dataset already exists. Skipping download."
    exit 0
fi

mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/data.tar
tar -xf data/data.tar -C data/
rm data/data.tar

echo "MMLU dataset downloaded and extracted successfully."
