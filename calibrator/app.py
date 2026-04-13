from cubindings import executeProgram

gridDim = 16
blockDim = 1024

result = executeProgram("""
    #include <iostream>
    #include <cuda_runtime.h>
                        
    #define ITERATIONS 40000000

    __global__ void ptx_kernel()
    {
        int tid = threadIdx.x;
        int tmp = tid;

        // Repeat the instruction in a C loop
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
    enable_metrics=True
)

print("Path", result.path)
print("Output", result.power_metric_result.total_energy_j)
print("GPU Duration", result.power_metric_result.kernel_duration_gpu_ns)
print("Exported X", result.exports["x"])