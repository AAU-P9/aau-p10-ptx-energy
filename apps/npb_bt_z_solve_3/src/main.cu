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

__global__ void bt_kernel(double* rhs_device,
		double* lhsA_device,
		double* lhsB_device,  
		double* lhsC_device){
	extern __shared__ double tmp_l_lhs[];
	double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];

	int t_j = blockDim.y * blockIdx.y + threadIdx.y;
	int j = t_j / 5;
	int m = t_j % 5;
	int i = blockDim.x * blockIdx.x + threadIdx.x+1;
	int l_i = threadIdx.x;
	if (j+1 < 1 || j+1 > JMAX - 2 || j >= PROBLEM_SIZE || i > IMAX - 2 ){return;}
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {

	j++;

	int k, n, p, ksize;

#define rhs(a, b, c, d) rhs_device[(((a) * (JMAXP+1) + (b)) * (IMAXP+1) + (c)) * 5 + (d)]

#define lhsA(a, b, c, d, e) lhsA_device[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * (JMAXP+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) lhsB_device[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * (JMAXP+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) lhsC_device[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * (JMAXP+1) + (d)) * (PROBLEM_SIZE-1) + (e)]

	double (*tmp2_l_lhs)[3][5][5] = (double(*)[3][5][5])tmp_l_lhs;
	double (*l_lhs)[5][5] = tmp2_l_lhs[l_i];
	double (*tmp2_l_r)[2][5] = (double(*)[2][5])tmp_l_r;
	double (*l_r)[5] = tmp2_l_r[l_i]; 

	double pivot, coeff;

	ksize = KMAX - 1;

	/*
	 * ---------------------------------------------------------------------
	 * compute the indices for storing the block-diagonal matrix;
	 * determine c (labeled f) and s jacobians   
	 * ---------------------------------------------------------------------
	 * performs guaussian elimination on this cell.
	 * ---------------------------------------------------------------------
	 * assumes that unpacking routines for non-first cells 
	 * preload C' and rhs' from previous cell.
	 * ---------------------------------------------------------------------
	 * assumed send happens outside this routine, but that
	 * c'(KMAX) and rhs'(KMAX) will be sent to next cell.
	 * ---------------------------------------------------------------------
	 * outer most do loops - sweeping in i direction
	 * ---------------------------------------------------------------------
	 */

	/* load data */
	#pragma unroll
	META_LOOP(p_pivot, 5, 5, true);
	for(p=0; p<5; p++){
		l_lhs[BB][p][m] = lhsB(p, m, 0, j, i-1);
		l_lhs[CC][p][m] = lhsC(p, m, 0, j, i-1);
	}

	l_r[1][m] = rhs(0, j, i, m);

	__syncthreads();

	/*
	 * ---------------------------------------------------------------------
	 * multiply c[0][j][i] by b_inverse and copy back to c
	 * multiply rhs(0) by b_inverse(0) and copy to rhs
	 * ---------------------------------------------------------------------
	 */
	META_LOOP(p_pivot_1, 5, 5, true);
	for(p=0; p<5; p++){
		pivot = 1.00/l_lhs[BB][p][p];
		if(m>p && m<5){l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;}
		if(m<5){l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;}
		if(p==m){l_r[1][p] = l_r[1][p]*pivot;}

		__syncthreads();

		if(p != m){
			coeff = l_lhs[BB][p][m];
			#pragma unroll
			META_LOOP(n_update, 5, 5, true);
			for(n=p+1; n<5; n++){l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];}
			#pragma unroll
			META_LOOP(n_update_1, 5, 5, true);
			for(n=0; n<5; n++){l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];}
			l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
		}

		__syncthreads();
	}

	/* update data */	
	rhs(0, j, i, m) = l_r[1][m];

	/*
	 * ---------------------------------------------------------------------
	 * begin inner most do loop
	 * do all the elements of the cell unless last 
	 * ---------------------------------------------------------------------
	 */
	META_LOOP(k_sweep, 1, PROBLEM_SIZE, false);
	for(k=1; k<=ksize-1; k++){
		/* load data */
		#pragma unroll
		META_LOOP(n_update_2, 5, 5, true);
		for(n=0; n<5; n++){
			l_lhs[AA][n][m] = lhsA(n, m, k, j, i-1);
			l_lhs[BB][n][m] = lhsB(n, m, k, j, i-1);
		}
		l_r[0][m] = l_r[1][m];
		l_r[1][m] = rhs(k, j, i, m);

		__syncthreads();

		/*
		 * ---------------------------------------------------------------------
		 * subtract A*lhs_vector(k-1) from lhs_vector(k)
		 * 
		 * rhs(k) = rhs(k) - A*rhs(k-1)
		 * ---------------------------------------------------------------------
		 */
		l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
			- l_lhs[AA][1][m]*l_r[0][1]
			- l_lhs[AA][2][m]*l_r[0][2]
			- l_lhs[AA][3][m]*l_r[0][3]
			- l_lhs[AA][4][m]*l_r[0][4];

		/*
		 * ---------------------------------------------------------------------
		 * B(k) = B(k) - C(k-1)*A(k)
		 * matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
		 * ---------------------------------------------------------------------
		 */
		#pragma unroll
		META_LOOP(p_pivot_2, 5, 5, true);
		for(p=0; p<5; p++){
			l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
				- l_lhs[AA][1][p]*l_lhs[CC][m][1]
				- l_lhs[AA][2][p]*l_lhs[CC][m][2]
				- l_lhs[AA][3][p]*l_lhs[CC][m][3]
				- l_lhs[AA][4][p]*l_lhs[CC][m][4];

		}

		__syncthreads();

		/* load data */
		#pragma unroll
		META_LOOP(n_update_3, 5, 5, true);
		for(n=0; n<5; n++){l_lhs[CC][n][m] = lhsC(n, m, k, j, i-1);}

		__syncthreads();

		/*
		 * ---------------------------------------------------------------------
		 * multiply c[k][j][i] by b_inverse and copy back to c
		 * multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
		 * ---------------------------------------------------------------------
		 */
		META_LOOP(p_pivot_3, 5, 5, true);
		for(p=0; p<5; p++){
			pivot = 1.00/l_lhs[BB][p][p];
			if(m>p && m<5){l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;}
			if(m<5){l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;}
			if(p==m){l_r[1][p] = l_r[1][p]*pivot;}

			__syncthreads();

			if(p != m){
				coeff = l_lhs[BB][p][m];
				#pragma unroll
				META_LOOP(n_update_4, 5, 5, true);
				for(n=p+1; n<5; n++){l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];}
				#pragma unroll
				META_LOOP(n_update_5, 5, 5, true);
				for(n=0; n<5; n++){l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];}
				l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
			}

			__syncthreads();
		}

		/* update data */
		#pragma unroll
		META_LOOP(n_update_6, 5, 5, true);
		for(n=0; n<5; n++){
			lhsC(n, m, k, j, i-1) = l_lhs[CC][n][m];
		}
		rhs(k, j, i, m) = l_r[1][m];
	}

	/*
	 * ---------------------------------------------------------------------
	 * now finish up special cases for last cell
	 * ---------------------------------------------------------------------
	 */
	/* load data */
	#pragma unroll
	META_LOOP(n_update_7, 5, 5, true);
	for(n=0; n<5; n++){
		l_lhs[AA][n][m] = lhsA(n, m, k, j, i-1);
		l_lhs[BB][n][m] = lhsB(n, m, k, j, i-1);
	}
	l_r[0][m] = l_r[1][m];
	l_r[1][m] = rhs(k, j, i, m);

	__syncthreads();

	/*
	 * ---------------------------------------------------------------------
	 * rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
	 * ---------------------------------------------------------------------
	 */
	l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
		- l_lhs[AA][1][m]*l_r[0][1]
		- l_lhs[AA][2][m]*l_r[0][2]
		- l_lhs[AA][3][m]*l_r[0][3]
		- l_lhs[AA][4][m]*l_r[0][4];

	/*
	 * ---------------------------------------------------------------------
	 * B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
	 * matmul_sub(AA,i,j,ksize,c,CC,i,j,ksize-1,c,BB,i,j,ksize)
	 * ---------------------------------------------------------------------
	 */
	#pragma unroll
	META_LOOP(p_pivot_4, 5, 5, true);
	for(p=0; p<5; p++){
		l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
			- l_lhs[AA][1][p]*l_lhs[CC][m][1]
			- l_lhs[AA][2][p]*l_lhs[CC][m][2]
			- l_lhs[AA][3][p]*l_lhs[CC][m][3]
			- l_lhs[AA][4][p]*l_lhs[CC][m][4];

	}

	/*
	 * ---------------------------------------------------------------------
	 * multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
	 * ---------------------------------------------------------------------
	 */
	META_LOOP(p_pivot_5, 5, 5, true);
	for(p=0; p<5; p++){
		pivot = 1.00/l_lhs[BB][p][p];
		if(m>p && m<5){l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;}
		if(p==m){l_r[1][p] = l_r[1][p]*pivot;}

		__syncthreads();

		if(p != m){
			coeff = l_lhs[BB][p][m];
			#pragma unroll
			META_LOOP(n_update_8, 5, 5, true);
			for(n=p+1; n<5; n++){l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];}
			l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
		}

		__syncthreads();
	}

	/* update data */
	rhs(k, j, i, m) = l_r[1][m];

	__syncthreads();

	/*
	 * ---------------------------------------------------------------------
	 * back solve: if last cell, then generate U(ksize)=rhs(ksize)
	 * else assume U(ksize) is loaded in un pack backsub_info
	 * so just use it
	 * after u(kstart) will be sent to next cell
	 * ---------------------------------------------------------------------
	 */
	META_LOOP(k_sweep_back, 1, PROBLEM_SIZE, false);
	for(k=ksize-1; k>=0; k--){
		#pragma unroll
		META_LOOP(n_update_9, M_SIZE, M_SIZE, true);
		for(n=0; n<M_SIZE; n++){
			rhs(k, j, i, m) = rhs(k, j, i, m) - lhsC(n, m, k, j, i-1)*rhs(k+1, j, i, n);
		}

		__syncthreads();
	}
#undef rhs
#undef lhsA
#undef lhsB
#undef lhsC
	}
}


int main() {
    METRICS_KERNEL_START

    double *rhs, *lhsA, *lhsB, *lhsC;
    cudaMalloc(&rhs,  BUF_5D);  cudaMemset(rhs,  0, BUF_5D);
    cudaMalloc(&lhsA, BUF_LHS); cudaMemset(lhsA, 0, BUF_LHS);
    cudaMalloc(&lhsB, BUF_LHS); cudaMemset(lhsB, 0, BUF_LHS);
    cudaMalloc(&lhsC, BUF_LHS); cudaMemset(lhsC, 0, BUF_LHS);

    size_t tpb_i = 4;
    size_t wx = round_work(IMAX-2, tpb_i);
    size_t wy = round_work(PROBLEM_SIZE*5, 5);
    dim3 block(wx/tpb_i, wy/5, 1);
    dim3 thread(tpb_i, 5, 1);
    size_t smem = sizeof(double) * tpb_i * (3*5*5 + 2*5);

    printf("[LOG] bt_z_solve_3: PROBLEM_SIZE=%d, ITERATIONS=%d\n", PROBLEM_SIZE, ITERATIONS);
    bt_kernel<<<block, thread, smem>>>(rhs, lhsA, lhsB, lhsC);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)block.x);
    EXPORT_N("gridDim_y", (int)block.y);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", (int)thread.x);
    EXPORT_N("blockDim_y", (int)thread.y);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(rhs); cudaFree(lhsA); cudaFree(lhsB); cudaFree(lhsC);
    return 0;
}
