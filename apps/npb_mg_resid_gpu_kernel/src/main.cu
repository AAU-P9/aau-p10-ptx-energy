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

__global__ void mg_kernel(double* u,
		double* v,
		double* r,
		double* a,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {

	double* u1=(double*)(extern_share_data);
	double* u2=(double*)(u1+M);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int lid=threadIdx.x;
	int i1;	

	META_LOOP(i1_loop, 1, MG_PROBLEM_SIZE, false);
	for(i1=lid; i1<n1; i1+=blockDim.x){
		u1[i1]=u[i3*n2*n1+(i2-1)*n1+i1]
			+u[i3*n2*n1+(i2+1)*n1+i1]
			+u[(i3-1)*n2*n1+i2*n1+i1]
			+u[(i3+1)*n2*n1+i2*n1+i1];
		u2[i1]=u[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} __syncthreads();
	META_LOOP(i1_loop_1, 1, MG_PROBLEM_SIZE, false);
	for(i1=lid+1; i1<n1-1; i1+=blockDim.x){
		r[i3*n2*n1+i2*n1+i1]=v[i3*n2*n1+i2*n1+i1]
			-a[0]*u[i3*n2*n1+i2*n1+i1]
			-a[2]*(u2[i1]+u1[i1-1]+u1[i1+1])
			-a[3]*(u2[i1-1]+u2[i1+1] );
	}
	}
}

int main() {
    METRICS_KERNEL_START

    double *u; cudaMalloc(&u, NV*sizeof(double)); cudaMemset(u, 0, NV*sizeof(double));
    double *v; cudaMalloc(&v, NV*sizeof(double)); cudaMemset(v, 0, NV*sizeof(double));
    double *r; cudaMalloc(&r, NV*sizeof(double)); cudaMemset(r, 0, NV*sizeof(double));
    double *a; cudaMalloc(&a, 4*sizeof(double)); cudaMemset(a, 0, 4*sizeof(double));
    int tpb_val = (NM < TPB) ? NM : TPB;
    int aw = (NM-2)*tpb_val;
    printf("[LOG] mg_resid_gpu_kernel: NM=%d M=%d ITERATIONS=%d\n", NM, M, ITERATIONS);
    dim3 grid(NM-2, NM-2);
    dim3 threads(tpb_val, 1);
    size_t smem = (size_t)2*M*sizeof(double);
    mg_kernel<<<grid, threads, smem>>>(u, v, r, a, NM, NM, NM, aw);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  grid.x);
    EXPORT_N("gridDim_y",  grid.y);
    EXPORT_N("gridDim_z",  grid.z);
    EXPORT_N("blockDim_x", threads.x);
    EXPORT_N("blockDim_y", threads.y);
    EXPORT_N("blockDim_z", threads.z);

    METRICS_KERNEL_END

    cudaFree(u); cudaFree(v); cudaFree(r); cudaFree(a);
    return 0;
}
