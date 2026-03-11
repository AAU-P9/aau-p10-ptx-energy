cd build
cmake ..
cmake --build . && cat ../matrix_mult.cu |./injector > modified_kernel.cu
clang++ -DUSE_LLI -S -emit-llvm modified_kernel.cu --no-cuda-version-check && lli modified_kernel.ll

