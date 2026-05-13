#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

// MG_PROBLEM_SIZE = finest-level grid side length (must be a power of 2)
// 32=class S, 128=class W, 256=class A
#ifndef MG_PROBLEM_SIZE
#define MG_PROBLEM_SIZE 32
#endif

// NM = 2 + MG_PROBLEM_SIZE (finest-level dimension including ghost cells)
#define NM  (2 + MG_PROBLEM_SIZE)
// M  = NM + 1 (scratch array size used in kernel bodies)
#define M   (NM + 1)
// NC = coarser level dimension
#define NC  (MG_PROBLEM_SIZE / 2 + 2)

#define NV  ((size_t)NM * NM * NM)
// NR >= NV; safe over-allocation for multigrid level offsets
#define NR  (NV * 2)

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif
#ifndef TPB
#define TPB 32
#endif

extern __shared__ double extern_share_data[];

__global__ void mg_kernel(double* z, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work){
	int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(thread_id>=(n1*n2*n3)){return;}
	z[thread_id]=0.0;
}

int main() {
    METRICS_KERNEL_START

    double *z; cudaMalloc(&z, NV*sizeof(double)); cudaMemset(z, 0, NV*sizeof(double));

    printf("[LOG] mg_zero3_gpu_kernel: NM=%d M=%d ITERATIONS=%d\n", NM, M, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        int tpb = TPB;
        int aw = NM*NM*NM;
        int grid = (aw + tpb - 1) / tpb;
        mg_kernel<<<grid, tpb>>>(z, NM, NM, NM, aw);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(z);
    return 0;
}
