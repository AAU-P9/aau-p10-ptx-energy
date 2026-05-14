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

__device__ void y_solve_gpu_device_fjac(double fjac[5][5], 
		double t_u[5],
		double rho_i, 
		double square, 
		double qs){
	double tmp1, tmp2;

	tmp1 = rho_i;
	tmp2 = tmp1 * tmp1;

	fjac[0][0] = 0.0;
	fjac[1][0] = 0.0;
	fjac[2][0] = 1.0;
	fjac[3][0] = 0.0;
	fjac[4][0] = 0.0;

	fjac[0][1] = - ( t_u[1]*t_u[2] ) * tmp2;
	fjac[1][1] = t_u[2] * tmp1;
	fjac[2][1] = t_u[1] * tmp1;
	fjac[3][1] = 0.0;
	fjac[4][1] = 0.0;

	fjac[0][2] = - ( t_u[2]*t_u[2]*tmp2) + constants_device::c2 * qs;
	fjac[1][2] = - constants_device::c2 *  t_u[1] * tmp1;
	fjac[2][2] = ( 2.0 - constants_device::c2 ) *  t_u[2] * tmp1;
	fjac[3][2] = - constants_device::c2 * t_u[3] * tmp1;
	fjac[4][2] = constants_device::c2;

	fjac[0][3] = - ( t_u[2]*t_u[3] ) * tmp2;
	fjac[1][3] = 0.0;
	fjac[2][3] = t_u[3] * tmp1;
	fjac[3][3] = t_u[2] * tmp1;
	fjac[4][3] = 0.0;

	fjac[0][4] = ( constants_device::c2 * 2.0 * square - constants_device::c1 * t_u[4] ) * t_u[2] * tmp2;
	fjac[1][4] = - constants_device::c2 * t_u[1]*t_u[2] * tmp2;
	fjac[2][4] = constants_device::c1 * t_u[4] * tmp1 - constants_device::c2 * ( qs + t_u[2]*t_u[2] * tmp2 );
	fjac[3][4] = - constants_device::c2 * ( t_u[2]*t_u[3] ) * tmp2;
	fjac[4][4] = constants_device::c1 * t_u[2] * tmp1;
}

__device__ void y_solve_gpu_device_njac(double njac[5][5], 
		double t_u[5],
		double rho_i){ 
	double tmp1, tmp2, tmp3;

	tmp1 = rho_i;
	tmp2 = tmp1 * tmp1;
	tmp3 = tmp1 * tmp2;

	njac[0][0] = 0.0;
	njac[1][0] = 0.0;
	njac[2][0] = 0.0;
	njac[3][0] = 0.0;
	njac[4][0] = 0.0;

	njac[0][1] = - constants_device::c3c4 * tmp2 * t_u[1];
	njac[1][1] = constants_device::c3c4 * tmp1;
	njac[2][1] = 0.0;
	njac[3][1] = 0.0;
	njac[4][1] = 0.0;

	njac[0][2] = - constants_device::con43 * constants_device::c3c4 * tmp2 * t_u[2];
	njac[1][2] = 0.0;
	njac[2][2] = constants_device::con43 * constants_device::c3c4 * tmp1;
	njac[3][2] = 0.0;
	njac[4][2] = 0.0;

	njac[0][3] = - constants_device::c3c4 * tmp2 * t_u[3];
	njac[1][3] = 0.0;
	njac[2][3] = 0.0;
	njac[3][3] = constants_device::c3c4 * tmp1;
	njac[4][3] = 0.0;

	njac[0][4] = - (  constants_device::c3c4
			- constants_device::c1345 ) * tmp3 * (t_u[1]*t_u[1])
		- ( constants_device::con43 * constants_device::c3c4
				- constants_device::c1345 ) * tmp3 * (t_u[2]*t_u[2])
		- ( constants_device::c3c4 - constants_device::c1345 ) * tmp3 * (t_u[3]*t_u[3])
		- constants_device::c1345 * tmp2 * t_u[4];

	njac[1][4] = ( constants_device::c3c4 - constants_device::c1345 ) * tmp2 * t_u[1];
	njac[2][4] = ( constants_device::con43 * constants_device::c3c4 - constants_device::c1345 ) * tmp2 * t_u[2];
	njac[3][4] = ( constants_device::c3c4 - constants_device::c1345 ) * tmp2 * t_u[3];
	njac[4][4] = ( constants_device::c1345 ) * tmp1;
}

__global__ void bt_kernel(double* qs_device, 
		double* rho_i_device,
		double* square_device, 
		double* u_device, 
		double* lhsA_device, 
		double* lhsB_device, 
		double* lhsC_device){
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if(k+0 < 1 || k+0 > KMAX-2 || k >= PROBLEM_SIZE  || j > JMAX-2 || i > IMAX-2){return;}

	int m;
	double tmp1, tmp2;

	double (*qs)[JMAXP+1][IMAXP+1]	= (double(*)[JMAXP+1][IMAXP+1]) qs_device;
	double (*rho_i)[JMAXP+1][IMAXP+1] = (double(*)[JMAXP+1][IMAXP+1]) rho_i_device;
	double (*square)[JMAXP+1][IMAXP+1]= (double(*)[JMAXP+1][IMAXP+1]) square_device; 
	double (*u)[JMAXP+1][IMAXP+1][5] = (double(*)[JMAXP+1][IMAXP+1][5]) u_device;

#define lhsA(a, b, c, d, e) lhsA_device[((((a) * 5 + (b)) *  PROBLEM_SIZE + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) lhsB_device[((((a) * 5 + (b)) *  PROBLEM_SIZE + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) lhsC_device[((((a) * 5 + (b)) *  PROBLEM_SIZE + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]

	double fjac[5][5], njac[5][5];

	double t_u[5];

	/*
	 * ---------------------------------------------------------------------
	 * this function computes the left hand side for the three y-factors   
	 * ---------------------------------------------------------------------
	 * compute the indices for storing the tri-diagonal matrix;
	 * determine a (labeled f) and n jacobians for cell c
	 * ---------------------------------------------------------------------
	 */
	tmp1 = constants_device::dt * constants_device::ty1;
	tmp2 = constants_device::dt * constants_device::ty2;

	for(m=0; m<5; m++){t_u[m] = u[k][j-1][i][m];}
	y_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j-1][i], square[k][j-1][i], qs[k][j-1][i]);
	y_solve_gpu_device_njac(njac, t_u, rho_i[k][j-1][i]);

	lhsA(0, 0, k, j, i-1) = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dy1; 
	lhsA(1, 0, k, j, i-1) = - tmp2 * fjac[1][0] - tmp1 * njac[1][0];
	lhsA(2, 0, k, j, i-1) = - tmp2 * fjac[2][0] - tmp1 * njac[2][0];
	lhsA(3, 0, k, j, i-1) = - tmp2 * fjac[3][0] - tmp1 * njac[3][0];
	lhsA(4, 0, k, j, i-1) = - tmp2 * fjac[4][0] - tmp1 * njac[4][0];

	lhsA(0, 1, k, j, i-1) = - tmp2 * fjac[0][1] - tmp1 * njac[0][1];
	lhsA(1, 1, k, j, i-1) = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dy2;
	lhsA(2, 1, k, j, i-1) = - tmp2 * fjac[2][1] - tmp1 * njac[2][1];
	lhsA(3, 1, k, j, i-1) = - tmp2 * fjac[3][1] - tmp1 * njac[3][1];
	lhsA(4, 1, k, j, i-1) = - tmp2 * fjac[4][1] - tmp1 * njac[4][1];

	lhsA(0, 2, k, j, i-1) = - tmp2 * fjac[0][2] - tmp1 * njac[0][2];
	lhsA(1, 2, k, j, i-1) = - tmp2 * fjac[1][2] - tmp1 * njac[1][2];
	lhsA(2, 2, k, j, i-1) = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dy3;
	lhsA(3, 2, k, j, i-1) = - tmp2 * fjac[3][2] - tmp1 * njac[3][2];
	lhsA(4, 2, k, j, i-1) = - tmp2 * fjac[4][2] - tmp1 * njac[4][2];

	lhsA(0, 3, k, j, i-1) = - tmp2 * fjac[0][3] - tmp1 * njac[0][3];
	lhsA(1, 3, k, j, i-1) = - tmp2 * fjac[1][3] - tmp1 * njac[1][3];
	lhsA(2, 3, k, j, i-1) = - tmp2 * fjac[2][3] - tmp1 * njac[2][3];
	lhsA(3, 3, k, j, i-1) = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dy4;
	lhsA(4, 3, k, j, i-1) = - tmp2 * fjac[4][3] - tmp1 * njac[4][3];

	lhsA(0, 4, k, j, i-1) = - tmp2 * fjac[0][4] - tmp1 * njac[0][4];
	lhsA(1, 4, k, j, i-1) = - tmp2 * fjac[1][4] - tmp1 * njac[1][4];
	lhsA(2, 4, k, j, i-1) = - tmp2 * fjac[2][4] - tmp1 * njac[2][4];
	lhsA(3, 4, k, j, i-1) = - tmp2 * fjac[3][4] - tmp1 * njac[3][4];
	lhsA(4, 4, k, j, i-1) = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dy5;

	for(m=0; m<5; m++){t_u[m] = u[k][j][i][m];}
	y_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i]);

	lhsB(0, 0, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * constants_device::dy1;
	lhsB(1, 0, k, j, i-1) = tmp1 * 2.0 * njac[1][0];
	lhsB(2, 0, k, j, i-1) = tmp1 * 2.0 * njac[2][0];
	lhsB(3, 0, k, j, i-1) = tmp1 * 2.0 * njac[3][0];
	lhsB(4, 0, k, j, i-1) = tmp1 * 2.0 * njac[4][0];

	lhsB(0, 1, k, j, i-1) = tmp1 * 2.0 * njac[0][1];
	lhsB(1, 1, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * constants_device::dy2;
	lhsB(2, 1, k, j, i-1) = tmp1 * 2.0 * njac[2][1];
	lhsB(3, 1, k, j, i-1) = tmp1 * 2.0 * njac[3][1];
	lhsB(4, 1, k, j, i-1) = tmp1 * 2.0 * njac[4][1];

	lhsB(0, 2, k, j, i-1) = tmp1 * 2.0 * njac[0][2];
	lhsB(1, 2, k, j, i-1) = tmp1 * 2.0 * njac[1][2];
	lhsB(2, 2, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * constants_device::dy3;
	lhsB(3, 2, k, j, i-1) = tmp1 * 2.0 * njac[3][2];
	lhsB(4, 2, k, j, i-1) = tmp1 * 2.0 * njac[4][2];

	lhsB(0, 3, k, j, i-1) = tmp1 * 2.0 * njac[0][3];
	lhsB(1, 3, k, j, i-1) = tmp1 * 2.0 * njac[1][3];
	lhsB(2, 3, k, j, i-1) = tmp1 * 2.0 * njac[2][3];
	lhsB(3, 3, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * constants_device::dy4;
	lhsB(4, 3, k, j, i-1) = tmp1 * 2.0 * njac[4][3];

	lhsB(0, 4, k, j, i-1) = tmp1 * 2.0 * njac[0][4];
	lhsB(1, 4, k, j, i-1) = tmp1 * 2.0 * njac[1][4];
	lhsB(2, 4, k, j, i-1) = tmp1 * 2.0 * njac[2][4];
	lhsB(3, 4, k, j, i-1) = tmp1 * 2.0 * njac[3][4];
	lhsB(4, 4, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * constants_device::dy5;

	for(m=0; m<5; m++){t_u[m] = u[k][j+1][i][m];}

	y_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j+1][i], square[k][j+1][i], qs[k][j+1][i]);
	y_solve_gpu_device_njac(njac, t_u, rho_i[k][j+1][i]);

	lhsC(0, 0, k, j, i-1) =  tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dy1;
	lhsC(1, 0, k, j, i-1) =  tmp2 * fjac[1][0] - tmp1 * njac[1][0];
	lhsC(2, 0, k, j, i-1) =  tmp2 * fjac[2][0] - tmp1 * njac[2][0];
	lhsC(3, 0, k, j, i-1) =  tmp2 * fjac[3][0] - tmp1 * njac[3][0];
	lhsC(4, 0, k, j, i-1) =  tmp2 * fjac[4][0] - tmp1 * njac[4][0];

	lhsC(0, 1, k, j, i-1) =  tmp2 * fjac[0][1] - tmp1 * njac[0][1];
	lhsC(1, 1, k, j, i-1) =  tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dy2;
	lhsC(2, 1, k, j, i-1) =  tmp2 * fjac[2][1] - tmp1 * njac[2][1];
	lhsC(3, 1, k, j, i-1) =  tmp2 * fjac[3][1] - tmp1 * njac[3][1];
	lhsC(4, 1, k, j, i-1) =  tmp2 * fjac[4][1] - tmp1 * njac[4][1];

	lhsC(0, 2, k, j, i-1) =  tmp2 * fjac[0][2] - tmp1 * njac[0][2];
	lhsC(1, 2, k, j, i-1) =  tmp2 * fjac[1][2] - tmp1 * njac[1][2];
	lhsC(2, 2, k, j, i-1) =  tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dy3;
	lhsC(3, 2, k, j, i-1) =  tmp2 * fjac[3][2] - tmp1 * njac[3][2];
	lhsC(4, 2, k, j, i-1) =  tmp2 * fjac[4][2] - tmp1 * njac[4][2];

	lhsC(0, 3, k, j, i-1) =  tmp2 * fjac[0][3] - tmp1 * njac[0][3];
	lhsC(1, 3, k, j, i-1) =  tmp2 * fjac[1][3] - tmp1 * njac[1][3];
	lhsC(2, 3, k, j, i-1) =  tmp2 * fjac[2][3] - tmp1 * njac[2][3];
	lhsC(3, 3, k, j, i-1) =  tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dy4;
	lhsC(4, 3, k, j, i-1) =  tmp2 * fjac[4][3] - tmp1 * njac[4][3];

	lhsC(0, 4, k, j, i-1) =  tmp2 * fjac[0][4] - tmp1 * njac[0][4];
	lhsC(1, 4, k, j, i-1) =  tmp2 * fjac[1][4] - tmp1 * njac[1][4];
	lhsC(2, 4, k, j, i-1) =  tmp2 * fjac[2][4] - tmp1 * njac[2][4];
	lhsC(3, 4, k, j, i-1) =  tmp2 * fjac[3][4] - tmp1 * njac[3][4];
	lhsC(4, 4, k, j, i-1) =  tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dy5;

#undef lhsA
#undef lhsB
#undef lhsC
}


int main() {
    METRICS_KERNEL_START

    double *qs, *rho_i, *square, *u, *lhsA, *lhsB, *lhsC;
    cudaMalloc(&qs,     BUF_3D); cudaMemset(qs,     0, BUF_3D);
    cudaMalloc(&rho_i,  BUF_3D); cudaMemset(rho_i,  0, BUF_3D);
    cudaMalloc(&square, BUF_3D); cudaMemset(square, 0, BUF_3D);
    cudaMalloc(&u,      BUF_5D); cudaMemset(u,      0, BUF_5D);
    cudaMalloc(&lhsA,   BUF_LHS); cudaMemset(lhsA,  0, BUF_LHS);
    cudaMalloc(&lhsB,   BUF_LHS); cudaMemset(lhsB,  0, BUF_LHS);
    cudaMalloc(&lhsC,   BUF_LHS); cudaMemset(lhsC,  0, BUF_LHS);

    size_t tpb = 32;
    size_t wx = round_work(IMAX-2, tpb);
    size_t wy = round_work(JMAX-2, 1);
    size_t wz = round_work(PROBLEM_SIZE, 1);
    dim3 block(wx/tpb, wy, wz);
    dim3 thread(tpb, 1, 1);

    printf("[LOG] bt_y_solve_2: PROBLEM_SIZE=%d, ITERATIONS=%d\n", PROBLEM_SIZE, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        bt_kernel<<<block, thread>>>(qs, rho_i, square, u, lhsA, lhsB, lhsC);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)block.x);
    EXPORT_N("gridDim_y", (int)block.y);
    EXPORT_N("gridDim_z", (int)block.z);
    EXPORT_N("blockDim_x", (int)thread.x);
    EXPORT_N("blockDim_y", (int)thread.y);
    EXPORT_N("blockDim_z", (int)thread.z);

    METRICS_KERNEL_END

    cudaFree(qs); cudaFree(rho_i); cudaFree(square); cudaFree(u);
    cudaFree(lhsA); cudaFree(lhsB); cudaFree(lhsC);
    return 0;
}
