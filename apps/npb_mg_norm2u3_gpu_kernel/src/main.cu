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

__global__ void mg_kernel(double* r,
		const int n1, 
		const int n2, 
		const int n3,
		double* res_sum,
		double* res_max,
		int number_of_blocks_on_x_axis,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {

	double* scratch_sum = (double*)(extern_share_data);
	double* scratch_max = (double*)(scratch_sum+blockDim.x);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int i1=threadIdx.x+1;

	double s=0.0;
	double my_rnmu=0.0;
	double a;

	META_LOOP(while_loop, 1, PROBLEM_SIZE, false);
	while(i1<n1-1){
		double r321=r[i3*n2*n1+i2*n1+i1];
		s=s+r321*r321;
		a=fabs(r321);
		my_rnmu=(a>my_rnmu)?a:my_rnmu;
		i1+=blockDim.x;
	}

	int lid=threadIdx.x;
	scratch_sum[lid]=s;
	scratch_max[lid]=my_rnmu;

	__syncthreads();
	META_LOOP(i_sweep_back, 1, PROBLEM_SIZE, false);
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(lid<i){
			scratch_sum[lid]+=scratch_sum[lid+i];
			scratch_max[lid]=(scratch_max[lid]>scratch_max[lid+i])?scratch_max[lid]:scratch_max[lid+i];
		}
		__syncthreads();
	}
	if(lid == 0){
		int idx=blockIdx.y*number_of_blocks_on_x_axis+blockIdx.x;
		res_sum[idx]=scratch_sum[0];
		res_max[idx]=scratch_max[0];
	}
	}
}

int main() {
    METRICS_KERNEL_START

    double *r; cudaMalloc(&r, NV*sizeof(double)); cudaMemset(r, 0, NV*sizeof(double));
    int blocks_x = NM-2;
    int aw = (NM-2)*TPB;
    int temp_size = blocks_x * (NM-2);
    double *res_sum, *res_max;
    cudaMalloc(&res_sum, temp_size*sizeof(double)); cudaMemset(res_sum, 0, temp_size*sizeof(double));
    cudaMalloc(&res_max, temp_size*sizeof(double)); cudaMemset(res_max, 0, temp_size*sizeof(double));
    printf("[LOG] mg_norm2u3_gpu_kernel: NM=%d M=%d ITERATIONS=%d\n", NM, M, ITERATIONS);
    dim3 grid(blocks_x, NM-2);
    dim3 threads(TPB, 1);
    size_t smem = (size_t)2*TPB*sizeof(double);
    mg_kernel<<<grid, threads, smem>>>(r, NM, NM, NM, res_sum, res_max, blocks_x, aw);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(r); cudaFree(res_sum); cudaFree(res_max);
    return 0;
}
