#!/bin/bash

ARCH=sm_89

echo "Generating LLVM IR..."
clang++ -x cuda -S -emit-llvm main.cu --cuda-gpu-arch=$ARCH

echo "Generating PTX..."
nvcc -ptx main.cu -o main.ptx
