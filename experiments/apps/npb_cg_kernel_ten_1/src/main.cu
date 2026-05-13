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

__global__ void bt_kernel(double* norm_temp, 
		double x[], 
		double z[]){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	double* share_data = (double*)extern_share_data;	  

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;	  

	share_data[threadIdx.x] = 0.0;

	if(thread_id >= NA){return;}	 

	share_data[threadIdx.x] = x[thread_id]*z[thread_id];

	__syncthreads();
	META_LOOP(i_sweep_back, 1, PROBLEM_SIZE, false);
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){norm_temp[blockIdx.x]=share_data[0];}
	}
}

int main() {
    METRICS_KERNEL_START

    double *norm_temp, *x, *z;
    cudaMalloc(&norm_temp, BUF_GDATA); cudaMemset(norm_temp, 0, BUF_GDATA);
    cudaMalloc(&x,         BUF_VEC);   cudaMemset(x,         0, BUF_VEC);
    cudaMalloc(&z,         BUF_VEC);   cudaMemset(z,         0, BUF_VEC);

    int grid = (NA + TPB - 1) / TPB;
    int thread = TPB;

    printf("[LOG] cg_kernel_ten_1: NA=%d, ITERATIONS=%d\n", NA, ITERATIONS);
    bt_kernel<<<grid, thread, TPB*sizeof(double)>>>(norm_temp, x, z);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)grid);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(norm_temp); cudaFree(x); cudaFree(z);
    return 0;
}
