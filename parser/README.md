mkdir build
cd build
cmake ..

cmake --build .

cmake --build . --target generate_llvm
cmake --build . --target parser
cmake --build . --target parser_ast


# Using lli
clang++ -DUSE_LLI -S -emit-llvm ../matrix_mult.cu --no-cuda-version-check && lli matrix_mult.ll

# Running on CUDA
nvcc -arch=sm_89 ../matrix_mult.cu -o matrix_mult && ./matrix_mult

# Compile with clang
clang++-20 ../matrix_mult.cu --cuda-gpu-arch=sm_89 --cuda-path=/usr/local/cuda-12 -L/usr/local/cuda-12/lib64 -lcudart

# LLVM-IR
clang++ ../matrix_mult.cu -S -emit-llvm --cuda-gpu-arch=sm_89 --cuda-path=/usr/local/cuda-12

# LLVM PTX
llc -march=nvptx64 -mattr=+ptx80 -mcpu=sm_89 matrix_mult-cuda-nvptx64-nvidia-cuda-sm_89.bc -o matrix_mult_llvm.ptxs

# NVCC PTX
nvcc -arch=sm_89 ../matrix_mult.cu --ptx -o matrix_mult_nvcc.ptx