#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>

#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 12
#endif

#define IMAX     (PROBLEM_SIZE)
#define JMAX     (PROBLEM_SIZE)
#define KMAX     (PROBLEM_SIZE)
#define IMAXP    (IMAX/2*2)
#define JMAXP    (JMAX/2*2)
#define M_SIZE   5
#define AA 0
#define BB 1
#define CC 2

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif

namespace constants_device {
__constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3,
    dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, dy5,
    dz1, dz2, dz3, dz4, dz5, dssp, dt,
    dxmax, dymax, dzmax, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5,
    dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1,
    yycon1, yycon2, yycon3, yycon4, yycon5,
    dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
    zzcon1, zzcon2, zzcon3, zzcon4, zzcon5,
    dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1,
    dIMAXm1, dJMAXm1, dKMAXm1,
    c1c2, c1c5, c3c4, c1345, coKMAX1,
    c1, c2, c3, c4, c5, c4dssp, c5dssp, dtdssp,
    dttx1, dttx2, dtty1, dtty2, dttz1, dttz2,
    c2dttx1, c2dtty1, c2dttz1,
    comz1, comz4, comz5, comz6,
    c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16,
    ce[5][13];
}

static inline size_t round_work(size_t n, size_t t) {
    return t == 0 ? n : ((n + t - 1) / t) * t;
}

#define BUF_5D  (sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5)
#define BUF_3D  (sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1))
#define BUF_LHS (sizeof(double)*(JMAXP+1)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5)

__global__ void bt_kernel(double* lhsA_device, 
		double* lhsB_device,
		double* lhsC_device){
	int t_j = blockDim.y * blockIdx.y + threadIdx.y;
	int j = t_j % PROBLEM_SIZE;
	int mn = t_j / PROBLEM_SIZE;
	int m = mn / 5;
	int n = mn % 5;
	int i = blockDim.x * blockIdx.x + threadIdx.x+1;

	if(j+1 < 1 || j+1 > JMAX-2 || j >= PROBLEM_SIZE || i > IMAX-2 || m >= 5){return;}

	j++;

	int ksize;

#define lhsA(a, b, c, d, e) lhsA_device[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * (JMAXP+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) lhsB_device[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * (JMAXP+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) lhsC_device[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * (JMAXP+1) + (d)) * (PROBLEM_SIZE-1) + (e)]

	ksize = KMAX - 1;

	/*
	 * ---------------------------------------------------------------------
	 * now jacobians set, so form left hand side in z direction
	 * ---------------------------------------------------------------------
	 */
	lhsA(m, n, 0, j, i-1) = 0.0;
	lhsB(m, n, 0, j, i-1) = (m==n)?1.0:0.0;
	lhsC(m, n, 0, j, i-1) = 0.0;

	lhsA(m, n, ksize, j, i-1) = 0.0;
	lhsB(m, n, ksize, j, i-1) = (m==n)?1.0:0.0;
	lhsC(m, n, ksize, j, i-1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC
}


int main() {
    METRICS_KERNEL_START

    double *lhsA, *lhsB, *lhsC;
    cudaMalloc(&lhsA, BUF_LHS); cudaMemset(lhsA, 0, BUF_LHS);
    cudaMalloc(&lhsB, BUF_LHS); cudaMemset(lhsB, 0, BUF_LHS);
    cudaMalloc(&lhsC, BUF_LHS); cudaMemset(lhsC, 0, BUF_LHS);

    size_t tpb = 32;
    size_t wx = round_work(IMAX-2, tpb);
    size_t wy = round_work(PROBLEM_SIZE*25, 1);
    dim3 block(wx/tpb, wy);
    dim3 thread(tpb, 1);

    printf("[LOG] bt_z_solve_1: PROBLEM_SIZE=%d, ITERATIONS=%d\n", PROBLEM_SIZE, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        bt_kernel<<<block, thread>>>(lhsA, lhsB, lhsC);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)block.x);
    EXPORT_N("gridDim_y", (int)block.y);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", (int)thread.x);
    EXPORT_N("blockDim_y", (int)thread.y);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(lhsA); cudaFree(lhsB); cudaFree(lhsC);
    return 0;
}
