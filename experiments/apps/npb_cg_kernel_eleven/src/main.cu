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

__global__ void bt_kernel(double norm_temp2, double x[], double z[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j >= NA){return;}
	x[j]=norm_temp2*z[j];
}

int main() {
    METRICS_KERNEL_START

    double norm_temp2 = 1.0;
    double *x, *z;
    cudaMalloc(&x, BUF_VEC); cudaMemset(x, 0, BUF_VEC);
    cudaMalloc(&z, BUF_VEC); cudaMemset(z, 0, BUF_VEC);

    int grid = (NA + TPB - 1) / TPB;
    int thread = TPB;

    printf("[LOG] cg_kernel_eleven: NA=%d, ITERATIONS=%d\n", NA, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        bt_kernel<<<grid, thread, 0>>>(norm_temp2, x, z);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)grid);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(x); cudaFree(z);
    return 0;
}
