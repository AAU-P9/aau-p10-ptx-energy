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
nvcc -arch=sm_89 matrix_mult.cu -o matrix_mult && ./matrix_mult