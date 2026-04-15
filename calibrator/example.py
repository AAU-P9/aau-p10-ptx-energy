from cubindings import execute_code
from analyser import run_analyser

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

used_energy = result.power_metric_result.total_energy_j

analyser_result = run_analyser(result.path, "/home/rasmus/aau-p10-ptx-energy/linear-model/weights.csv")
predicted_energy = analyser_result.predicted_energy_joules

print(f"Used energy: {used_energy:.6f} J"
      f"\nPredicted energy: {predicted_energy:.6f} J"
)

print(f"Difference: {abs(used_energy - predicted_energy):.6f} J"
      f"\nRelative error: {abs(used_energy - predicted_energy) / used_energy * 100:.2f}%"
)