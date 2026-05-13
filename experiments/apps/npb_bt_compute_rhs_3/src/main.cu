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

__global__ void bt_kernel(double* us_device,
		double* vs_device,
		double* ws_device,
		double* qs_device,
		double* rho_i_device,
		double* square_device,
		double* u_device,
		double* rhs_device){
	/* 1 <= k <= KMAX-2 */
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y+1;
	int i = blockIdx.x * blockDim.x + threadIdx.x+1;

	if(k+0 < 1 || k+0 > KMAX-2 || k >= KMAX || j > JMAX-2 || i > IMAX-2){return;}

	double uijk, up1, um1;

	double (*us)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1])us_device;
	double (*vs)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1])vs_device;  
	double (*ws)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1])ws_device;  
	double (*qs)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1])qs_device;  
	double (*rho_i)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1])rho_i_device; 
	double (*square)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1])square_device; 
	double (*u)[JMAXP+1][IMAXP+1][5] = (double(*)[JMAXP+1][IMAXP+1][5])u_device; 
	double (*rhs)[JMAXP+1][IMAXP+1][5] = (double(*)[JMAXP+1][IMAXP+1][5])rhs_device; 

	uijk = us[k][j][i];
	up1 = us[k][j][i+1];
	um1 = us[k][j][i-1];

	rhs[k][j][i][0] = rhs[k][j][i][0] + constants_device::dx1tx1 * 
		(u[k][j][i+1][0] - 2.0*u[k][j][i][0] + 
		 u[k][j][i-1][0]) -
		constants_device::tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]);

	rhs[k][j][i][1] = rhs[k][j][i][1] + constants_device::dx2tx1 * 
		(u[k][j][i+1][1] - 2.0*u[k][j][i][1] + 
		 u[k][j][i-1][1]) +
		constants_device::xxcon2*constants_device::con43 * (up1 - 2.0*uijk + um1) -
		constants_device::tx2 * (u[k][j][i+1][1]*up1 - 
				u[k][j][i-1][1]*um1 +
				(u[k][j][i+1][4]- square[k][j][i+1]-
				 u[k][j][i-1][4]+ square[k][j][i-1])*
				constants_device::c2);

	rhs[k][j][i][2] = rhs[k][j][i][2] + constants_device::dx3tx1 * 
		(u[k][j][i+1][2] - 2.0*u[k][j][i][2] +
		 u[k][j][i-1][2]) +
		constants_device::xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
				vs[k][j][i-1]) -
		constants_device::tx2 * (u[k][j][i+1][2]*up1 - 
				u[k][j][i-1][2]*um1);

	rhs[k][j][i][3] = rhs[k][j][i][3] + constants_device::dx4tx1 * 
		(u[k][j][i+1][3] - 2.0*u[k][j][i][3] +
		 u[k][j][i-1][3]) +
		constants_device::xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
				ws[k][j][i-1]) -
		constants_device::tx2 * (u[k][j][i+1][3]*up1 - 
				u[k][j][i-1][3]*um1);

	rhs[k][j][i][4] = rhs[k][j][i][4] + constants_device::dx5tx1 * 
		(u[k][j][i+1][4] - 2.0*u[k][j][i][4] +
		 u[k][j][i-1][4]) +
		constants_device::xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
				qs[k][j][i-1]) +
		constants_device::xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
				um1*um1) +
		constants_device::xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] - 
				2.0*u[k][j][i][4]*rho_i[k][j][i] +
				u[k][j][i-1][4]*rho_i[k][j][i-1]) -
		constants_device::tx2 * ( (constants_device::c1*u[k][j][i+1][4] - 
					constants_device::c2*square[k][j][i+1])*up1 -
				(constants_device::c1*u[k][j][i-1][4] - 
				 constants_device::c2*square[k][j][i-1])*um1 );
}


int main() {
    METRICS_KERNEL_START

    double *us, *vs, *ws, *qs, *rho_i, *square, *u, *rhs;
    cudaMalloc(&us,     BUF_3D); cudaMemset(us,     0, BUF_3D);
    cudaMalloc(&vs,     BUF_3D); cudaMemset(vs,     0, BUF_3D);
    cudaMalloc(&ws,     BUF_3D); cudaMemset(ws,     0, BUF_3D);
    cudaMalloc(&qs,     BUF_3D); cudaMemset(qs,     0, BUF_3D);
    cudaMalloc(&rho_i,  BUF_3D); cudaMemset(rho_i,  0, BUF_3D);
    cudaMalloc(&square, BUF_3D); cudaMemset(square, 0, BUF_3D);
    cudaMalloc(&u,      BUF_5D); cudaMemset(u,      0, BUF_5D);
    cudaMalloc(&rhs,    BUF_5D); cudaMemset(rhs,    0, BUF_5D);

    size_t tpb = 32;
    size_t wx = round_work(IMAX-2, tpb);
    size_t wy = round_work(JMAX-2, 1);
    size_t wz = round_work(KMAX, 1);
    dim3 block(wx/tpb, wy, wz);
    dim3 thread(tpb, 1, 1);

    printf("[LOG] bt_compute_rhs_3: PROBLEM_SIZE=%d, ITERATIONS=%d\n", PROBLEM_SIZE, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        bt_kernel<<<block, thread>>>(us, vs, ws, qs, rho_i, square, u, rhs);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)(wx/tpb));
    EXPORT_N("gridDim_y", (int)wy);
    EXPORT_N("gridDim_z", (int)wz);
    EXPORT_N("blockDim_x", (int)tpb);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(us); cudaFree(vs); cudaFree(ws); cudaFree(qs);
    cudaFree(rho_i); cudaFree(square); cudaFree(u); cudaFree(rhs);
    return 0;
}
