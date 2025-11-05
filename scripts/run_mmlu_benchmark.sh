#!/bin/bash

# This script runs the MMLU benchmark.
# You can override the model_path and other settings by passing them as command-line arguments.

python3 benchmark/run_mmlu_benchmark.py "$@"
