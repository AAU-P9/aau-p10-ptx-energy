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

__global__ void mg_kernel(double* z_device,
		double* u_device,
		int mm1, 
		int mm2, 
		int mm3,
		int n1, 
		int n2, 
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* z1 = (double*)(extern_share_data);
	double* z2 = (double*)(z1+M);
	double* z3 = (double*)(z2+M);

	int i3=blockIdx.y*blockDim.y+threadIdx.y;
	int i2=blockIdx.x;
	int i1=threadIdx.x;

	z1[i1]=z_device[i3*mm2*mm1+(i2+1)*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1];
	z2[i1]=z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1];
	z3[i1]=z_device[(i3+1)*mm2*mm1+(i2+1)*mm1+i1]+z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z1[i1];

	__syncthreads();
	if(i1<mm1-1){
		double z321=z_device[i3*mm2*mm1+i2*mm1+i1];
		u_device[2*i3*n2*n1+2*i2*n1+2*i1]+=z321;
		u_device[2*i3*n2*n1+2*i2*n1+2*i1+1]+=0.5*(z_device[i3*mm2*mm1+i2*mm1+i1+1]+z321);
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1]+=0.5*z1[i1];
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.25*(z1[i1]+z1[i1+1]);
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1]+=0.5*z2[i1];
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1]+=0.25*(z2[i1]+z2[i1+1]);
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1]+=0.25*z3[i1];
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.125*(z3[i1]+z3[i1+1]);
	}
}

int main() {
    METRICS_KERNEL_START

    double *z; cudaMalloc(&z, (size_t)NC*NC*NC*sizeof(double)); cudaMemset(z, 0, (size_t)NC*NC*NC*sizeof(double));
    double *u; cudaMalloc(&u, NV*sizeof(double)); cudaMemset(u, 0, NV*sizeof(double));
    int mm1=NC, mm2=NC, mm3=NC, n1=NM, n2=NM, n3=NM;
    printf("[LOG] mg_interp_gpu_kernel: NM=%d M=%d ITERATIONS=%d\n", NM, M, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        // tpb=mm1 to match blockDim.x used for scratch indexing
        dim3 tpb2(mm1, 1);
        int aw = (mm2-1)*mm1;
        dim3 grid(mm2-1, mm3-1);
        size_t smem = (size_t)3*M*sizeof(double);
        mg_kernel<<<grid, tpb2, smem>>>(z, u, mm1, mm2, mm3, n1, n2, n3, aw);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(z); cudaFree(u);
    return 0;
}
