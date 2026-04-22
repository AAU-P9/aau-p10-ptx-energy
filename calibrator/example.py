from cubindings import execute_code

gridDim = 64
blockDim = 1024

result = execute_code("""
    #include <iostream>
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include "ptx_meta.h"
                        
    #define ITERATIONS 40000000

    __global__ void ptx_kernel()
    {
        int tid = threadIdx.x;
        int tmp = tid;

        // Repeat the instruction in a C loop
        META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
        for(int i = 0; i < ITERATIONS; ++i)
        {
            asm volatile (     
                "mov.u32 %0, %0;\\n\\t"  // move tmp to tmp (self-move)
                : "+r"(tmp)             // %0 is a register mapped to tmp
            );
        }
    }
    int main() {
        int x = 5;
        x *= 2;

        ptx_kernel<<<""" + str(gridDim) + """, """ + str(blockDim) + """>>>();
        cudaDeviceSynchronize();

        EXPORT_N("x", x)

        return 0;
    }
    """,
    nvcc_args=[],
    binary_args=[],
    enable_metrics=True,
)