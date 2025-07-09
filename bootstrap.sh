#!/bin/bash
set -euo pipefail
source env.sh

CURRENT_DIR=$(pwd)
BUILD_DIR="build"

# Build script for a CMake project

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake $CURRENT_DIR
cd $CURRENT_DIR
