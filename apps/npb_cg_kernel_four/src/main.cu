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

__global__ void bt_kernel(double* d, 
		double* p, 
		double* q, 
		double global_data[]){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	double* share_data = (double*)extern_share_data; 

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	share_data[threadIdx.x] = p[thread_id] * q[thread_id];

	__syncthreads();
	META_LOOP(i_sweep_back, 5, 5, false);  // log2(TPB=32) reduction steps
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){global_data[blockIdx.x]=share_data[0];}
	}
}

int main() {
    METRICS_KERNEL_START

    double *d, *p, *q, *global_data;
    cudaMalloc(&d,           sizeof(double)); cudaMemset(d,           0, sizeof(double));
    cudaMalloc(&p,           BUF_VEC);   cudaMemset(p,           0, BUF_VEC);
    cudaMalloc(&q,           BUF_VEC);   cudaMemset(q,           0, BUF_VEC);
    cudaMalloc(&global_data, BUF_GDATA); cudaMemset(global_data, 0, BUF_GDATA);

    int grid = (NA + TPB - 1) / TPB;
    int thread = TPB;

    printf("[LOG] cg_kernel_four: NA=%d, ITERATIONS=%d\n", NA, ITERATIONS);
    bt_kernel<<<grid, thread, TPB*sizeof(double)>>>(d, p, q, global_data);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)grid);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(d); cudaFree(p); cudaFree(q); cudaFree(global_data);
    return 0;
}
