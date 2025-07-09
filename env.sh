#!/bin/bash
export CC=/opt/homebrew/opt/llvm@17/bin/clang
export CXX=/opt/homebrew/opt/llvm@17/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
