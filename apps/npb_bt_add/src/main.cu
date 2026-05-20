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

__global__ void bt_kernel(double * u_device, 
		double * rhs_device){
	int k = blockIdx.z * blockDim.z + threadIdx.z+1;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int t_i = blockIdx.x * blockDim.x + threadIdx.x;
	int i = t_i/5 + 1;
	int m = t_i%5;

	if(k > KMAX-2 || j+1 < 1 || j+1 > JMAX-2 || j >= PROBLEM_SIZE || i > IMAX-2){return;}
    j++;
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	u_device[((k * (JMAXP+1) + j) * (IMAXP+1) + i) * 5 + m]	+= rhs_device[((k * (JMAXP+1) + j) * (IMAXP+1) + i) * 5 + m];
	}
}


int main() {
    METRICS_KERNEL_START

    double *u_device, *rhs_device;
    cudaMalloc(&u_device,   BUF_5D);
    cudaMalloc(&rhs_device, BUF_5D);
    cudaMemset(u_device,   0, BUF_5D);
    cudaMemset(rhs_device, 0, BUF_5D);

    size_t tpb = 32;
    size_t wx = round_work((IMAX-2)*5, tpb);
    size_t wy = round_work(PROBLEM_SIZE, 1);
    size_t wz = round_work(KMAX-2, 1);
    dim3 block(wx/tpb, wy, wz);
    dim3 thread(tpb, 1, 1);

    printf("[LOG] bt_add: PROBLEM_SIZE=%d, ITERATIONS=%d\n", PROBLEM_SIZE, ITERATIONS);
    bt_kernel<<<block, thread>>>(u_device, rhs_device);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)(wx/tpb));
    EXPORT_N("gridDim_y", (int)wy);
    EXPORT_N("gridDim_z", (int)wz);
    EXPORT_N("blockDim_x", (int)tpb);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(u_device);
    cudaFree(rhs_device);
    return 0;
}
