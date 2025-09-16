#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CUDA_EXEC_DIR="$BASE_DIR/build/outputs/cuda-demos"
OUTPUT_DIR="$BASE_DIR/.profile"

for exec_file in "$CUDA_EXEC_DIR"/*; do
    if [[ -x "$exec_file" && ! -d "$exec_file" ]]; then
        exec_name=$(basename "$exec_file")
        profile_output_dir="$OUTPUT_DIR/$exec_name"
        mkdir -p "$profile_output_dir"
        
        echo "Profiling $exec_name..."
        ncu --set full --target-processes all --output "$profile_output_dir/$exec_name" "$exec_file"
        
        echo "Profile data for $exec_name saved in $profile_output_dir"
    fi
done
