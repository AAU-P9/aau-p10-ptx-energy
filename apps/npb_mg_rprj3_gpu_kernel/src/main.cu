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

__global__ void mg_kernel(double* r_device,
		double* s_device,
		int m1k,
		int m2k,
		int m3k,
		int m1j,
		int m2j,
		int m3j,
		int d1, 
		int d2, 
		int d3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {

	int j3,j2,j1,i3,i2,i1;
	double x2,y2;

	double* x1 = (double*)(extern_share_data);
	double* y1 = (double*)(x1+M);

	j3=blockIdx.y*blockDim.y+threadIdx.y+1;
	j2=blockIdx.x+1;
	j1=threadIdx.x+1;

	i3=2*j3-d3;
	i2=2*j2-d2;
	i1=2*j1-d1;
	x1[i1]=r_device[(i3+1)*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+1)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1];
	y1[i1]=r_device[i3*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+i2*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1];		
	__syncthreads();
	if(j1<m1j-1){
		i1=2*j1-d1;
		y2=r_device[i3*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+i2*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1+1];
		x2=r_device[(i3+1)*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+1)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1+1];
		s_device[j3*m2j*m1j+j2*m1j+j1]=
			0.5*r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+1]
			+0.25*(r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1]
					+r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+2]+x2)
			+0.125*(x1[i1]+x1[i1+2]+y2)
			+0.0625*(y1[i1]+y1[i1+2]);
	}
	}
}

int main() {
    METRICS_KERNEL_START

    double *r_dev; cudaMalloc(&r_dev, NV*sizeof(double)); cudaMemset(r_dev, 0, NV*sizeof(double));
    double *s_dev; cudaMalloc(&s_dev, (size_t)NC*NC*NC*sizeof(double)); cudaMemset(s_dev, 0, (size_t)NC*NC*NC*sizeof(double));
    int m1k=NM, m2k=NM, m3k=NM;
    int m1j=NC, m2j=NC, m3j=NC;
    int d1=1, d2=1, d3=1;
    int aw = (m2j-2)*(m1j-1);
    printf("[LOG] mg_rprj3_gpu_kernel: NM=%d M=%d ITERATIONS=%d\n", NM, M, ITERATIONS);
    dim3 tpb2(m1j-1, 1);
    dim3 grid(m2j-2, m3j-2);
    size_t smem = (size_t)2*M*sizeof(double);
    mg_kernel<<<grid, tpb2, smem>>>(r_dev, s_dev, m1k, m2k, m3k, m1j, m2j, m3j, d1, d2, d3, aw);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  grid.x);
    EXPORT_N("gridDim_y",  grid.y);
    EXPORT_N("gridDim_z",  grid.z);
    EXPORT_N("blockDim_x", tpb2.x);
    EXPORT_N("blockDim_y", tpb2.y);
    EXPORT_N("blockDim_z", tpb2.z);

    METRICS_KERNEL_END

    cudaFree(r_dev); cudaFree(s_dev);
    return 0;
}
