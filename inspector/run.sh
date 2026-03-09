cd build
cmake ..
cmake --build . && ./injector ../matrix_mult.cu > modified_kernel.cu
clang++ -DUSE_LLI -S -emit-llvm modified_kernel.cu --no-cuda-version-check && lli modified_kernel.ll

