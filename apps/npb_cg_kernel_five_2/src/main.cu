#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>

#ifndef NA
#define NA 1400
#endif

#ifndef NONZER
#define NONZER 7
#endif

#define NZ  (NA*(NONZER+1)*(NONZER+1))
#define NAZ (NA*(NONZER+1))

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif

#ifndef TPB
#define TPB 32
#endif

extern __shared__ double extern_share_data[];

// Buffer sizes
#define BUF_VEC   ((NA+2)*sizeof(double))
#define BUF_A     (NZ*sizeof(double))
#define BUF_CIDX  (NZ*sizeof(int))
#define BUF_ROW   ((NA+1)*sizeof(int))
#define BUF_GDATA (((NA+TPB-1)/TPB)*sizeof(double))

__global__ void bt_kernel(double alpha, 
		double* q, 
		double* r){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j >= NA){return;}
	r[j] -= alpha * q[j];
}

int main() {
    METRICS_KERNEL_START

    double alpha = 1.0;
    double *q, *r;
    cudaMalloc(&q, BUF_VEC); cudaMemset(q, 0, BUF_VEC);
    cudaMalloc(&r, BUF_VEC); cudaMemset(r, 0, BUF_VEC);

    int grid = (NA + TPB - 1) / TPB;
    int thread = TPB;

    printf("[LOG] cg_kernel_five_2: NA=%d, ITERATIONS=%d\n", NA, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        bt_kernel<<<grid, thread, 0>>>(alpha, q, r);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)grid);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(q); cudaFree(r);
    return 0;
}
