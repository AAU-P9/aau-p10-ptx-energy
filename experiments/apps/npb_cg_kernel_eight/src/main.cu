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

__global__ void bt_kernel(int colidx[], 
		int rowstr[], 
		double a[], 
		double r[], 
		double* z){
	double* share_data = (double*)extern_share_data;

	int j = (int) ((blockIdx.x*blockDim.x+threadIdx.x) / blockDim.x);
	int local_id = threadIdx.x;

	int begin = rowstr[j];
	int end = rowstr[j+1];
	double sum = 0.0;
	for(int k=begin+local_id; k<end; k+=blockDim.x){
		sum = sum + a[k]*z[colidx[k]];
	}
	share_data[local_id] = sum;

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){r[j]=share_data[0];}
}

int main() {
    METRICS_KERNEL_START

    int *colidx, *rowstr;
    double *a, *r, *z;
    cudaMalloc(&colidx, BUF_CIDX); cudaMemset(colidx, 0, BUF_CIDX);
    cudaMalloc(&rowstr, BUF_ROW);  cudaMemset(rowstr, 0, BUF_ROW);
    cudaMalloc(&a,      BUF_A);    cudaMemset(a,      0, BUF_A);
    cudaMalloc(&r,      BUF_VEC);  cudaMemset(r,      0, BUF_VEC);
    cudaMalloc(&z,      BUF_VEC);  cudaMemset(z,      0, BUF_VEC);

    int grid = NA;
    int thread = TPB;

    printf("[LOG] cg_kernel_eight: NA=%d, ITERATIONS=%d\n", NA, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        bt_kernel<<<grid, thread, TPB*sizeof(double)>>>(colidx, rowstr, a, r, z);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)grid);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(colidx); cudaFree(rowstr); cudaFree(a); cudaFree(r); cudaFree(z);
    return 0;
}
