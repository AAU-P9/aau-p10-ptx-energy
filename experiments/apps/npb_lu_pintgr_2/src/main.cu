#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

// PROBLEM_SIZE = ISIZ1=ISIZ2=ISIZ3 (cubic grid side length)
// 12=class S, 33=class W, 64=class A
#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 12
#endif

#define NX PROBLEM_SIZE
#define NY PROBLEM_SIZE
#define NZ PROBLEM_SIZE

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif
#ifndef TPB
#define TPB 32
#endif

// LU physical constants
#define C1  1.40e+00
#define C2  0.40e+00
#define C3  1.00e-01
#define C4  1.00e+00
#define C5  1.40e+00

// Access macros (m,i,j,k layout: [m + 5*(i + NX*(j + NY*k))])
#define u(m,i,j,k)     u    [(m)+5*((i)+NX*((j)+NY*(k)))]
#define v(m,i,j,k)     v    [(m)+5*((i)+NX*((j)+NY*(k)))]
#define rsd(m,i,j,k)   rsd  [(m)+5*((i)+NX*((j)+NY*(k)))]
#define frct(m,i,j,k)  frct [(m)+5*((i)+NX*((j)+NY*(k)))]
#define rho_i(i,j,k)   rho_i[(i)+NX*((j)+NY*(k))]
#define qs(i,j,k)      qs   [(i)+NX*((j)+NY*(k))]

extern __shared__ double extern_share_data[];

// Buffer sizes
#define BUF_5NXZ   ((size_t)5*NX*NY*NZ*sizeof(double))
#define BUF_NXZ    ((size_t)  NX*NY*NZ*sizeof(double))
// norm_buf_size = max(5*(NY-2)*(NZ-2), NX*NY, NX*NZ, NY*NZ)
#define NORM_BUF_SIZE (5*(NY-2)*(NZ-2) > NX*NY ? \
    (5*(NY-2)*(NZ-2) > NX*NZ ? (5*(NY-2)*(NZ-2) > NY*NZ ? 5*(NY-2)*(NZ-2) : NY*NZ) : \
     (NX*NZ > NY*NZ ? NX*NZ : NY*NZ)) : \
    (NX*NY > NX*NZ ? (NX*NY > NY*NZ ? NX*NY : NY*NZ) : (NX*NZ > NY*NZ ? NX*NZ : NY*NZ)))
#define BUF_NORM   ((size_t)NORM_BUF_SIZE*sizeof(double))

namespace constants_device {
    __constant__ double ce[13][5];
    __constant__ double dxi, deta, dzeta;
    __constant__ double tx1, tx2, tx3;
    __constant__ double ty1, ty2, ty3;
    __constant__ double tz1, tz2, tz3;
    __constant__ double dx1, dx2, dx3, dx4, dx5;
    __constant__ double dy1, dy2, dy3, dy4, dy5;
    __constant__ double dz1, dz2, dz3, dz4, dz5;
    __constant__ double dssp;
    __constant__ double dt, omega;
}

// Helper to initialise __constant__ symbols with plausible values
static void init_constants() {
    double ce[13][5] = {
        {2.0, 1.0, 2.0, 2.0, 1.0},
        {0.0, 0.0, 2.0, 2.0, 2.0},
        {0.0, 0.0, 0.0, 0.0, 2.0},
        {4.0, 0.0, 0.0, 0.0, 2.0},
        {5.0, 1.0, 0.0, 0.0, 1.0},
        {3.0, 2.0, 2.0, 2.0, 1.0},
        {0.5, 3.0, 3.0, 3.0, 1.0},
        {0.02,4.0, 4.0, 4.0, 1.0},
        {0.01,3.0, 3.0, 3.0, 1.0},
        {0.03,2.0, 2.0, 2.0, 1.0},
        {0.5, 0.5, 0.5, 0.5, 1.0},
        {0.4, 0.4, 0.4, 0.4, 1.0},
        {0.3, 0.3, 0.3, 0.3, 1.0},
    };
    double dxi=1.0/(NX-1), deta=1.0/(NY-1), dzeta=1.0/(NZ-1);
    double tx1=1.0/(dxi*dxi), tx2=1.0/(2.0*dxi), tx3=1.0/dxi;
    double ty1=1.0/(deta*deta), ty2=1.0/(2.0*deta), ty3=1.0/deta;
    double tz1=1.0/(dzeta*dzeta), tz2=1.0/(2.0*dzeta), tz3=1.0/dzeta;
    double dx1=0.75, dx2=0.75, dx3=0.75, dx4=0.75, dx5=0.75;
    double dy1=0.75, dy2=0.75, dy3=0.75, dy4=0.75, dy5=0.75;
    double dz1=0.75, dz2=0.75, dz3=0.75, dz4=0.75, dz5=0.75;
    double dssp=0.25*1.0, dt=0.5, omega=1.2;
    cudaMemcpyToSymbol(constants_device::ce,    ce,    sizeof(ce));
    cudaMemcpyToSymbol(constants_device::dxi,   &dxi,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::deta,  &deta,  sizeof(double));
    cudaMemcpyToSymbol(constants_device::dzeta, &dzeta, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tx1,   &tx1,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::tx2,   &tx2,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::tx3,   &tx3,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::ty1,   &ty1,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::ty2,   &ty2,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::ty3,   &ty3,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::tz1,   &tz1,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::tz2,   &tz2,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::tz3,   &tz3,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx1,   &dx1,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx2,   &dx2,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx3,   &dx3,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx4,   &dx4,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx5,   &dx5,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy1,   &dy1,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy2,   &dy2,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy3,   &dy3,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy4,   &dy4,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy5,   &dy5,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz1,   &dz1,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz2,   &dz2,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz3,   &dz3,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz4,   &dz4,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz5,   &dz5,   sizeof(double));
    cudaMemcpyToSymbol(constants_device::dssp,  &dssp,  sizeof(double));
    cudaMemcpyToSymbol(constants_device::dt,    &dt,    sizeof(double));
    cudaMemcpyToSymbol(constants_device::omega, &omega, sizeof(double));
}
__device__ static void exact_gpu_device(const int i,
		const int j,
		const int k,
		double* u000ijk,
		const int nx,
		const int ny,
		const int nz){
	int m;
	double xi, eta, zeta;
	using namespace constants_device;
	xi=(double)i/(double)(nx-1);
	eta=(double)j/(double)(ny-1);
	zeta=(double)k/(double)(nz-1);
	#pragma unroll
	META_LOOP(m_vars, 5, 5, true);
	for(m=0; m<5; m++){
		u000ijk[m]=ce[0][m]+
			(ce[1][m]+
			 (ce[4][m]+
			  (ce[7][m]+
			   ce[10][m]*xi)*xi)*xi)*xi+ 
			(ce[2][m]+
			 (ce[5][m]+
			  (ce[8][m]+
			   ce[11][m]*eta)*eta)*eta)*eta+ 
			(ce[3][m]+
			 (ce[6][m]+
			  (ce[9][m]+
			   ce[12][m]*zeta)*zeta)*zeta)*zeta;
	}
}

__global__ static void lu_kernel(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	int i, j, k, kp, ip;

	double* phi1 = (double*)extern_share_data;
	double* phi2 = (double*)phi1+(blockDim.x*blockDim.y);
	double* frc2 = (double*)phi2+(blockDim.x*blockDim.y);

	i=blockIdx.x*(blockDim.x-1)+1;
	k=blockIdx.y*(blockDim.x-1)+2;
	kp=threadIdx.y;
	ip=threadIdx.x;

	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * initialize
	 * ---------------------------------------------------------------------
	 */
	if(((k+kp)<(nz-1))&&((i+ip)<(nx-1))){
		j=1;
		phi1[kp+(ip*blockDim.x)]=C2*(u(4,i+ip,j,k+kp)-0.5*(u(1,i+ip,j,k+kp)*u(1,i+ip,j,k+kp)+u(2,i+ip,j,k+kp)*u(2,i+ip,j,k+kp)+u(3,i+ip,j,k+kp)*u(3,i+ip,j,k+kp))/u(0,i+ip,j,k+kp));
		j=ny-3;
		phi2[kp+(ip*blockDim.x)]=C2*(u(4,i+ip,j,k+kp)-0.5*(u(1,i+ip,j,k+kp)*u(1,i+ip,j,k+kp)+u(2,i+ip,j,k+kp)*u(2,i+ip,j,k+kp)+u(3,i+ip,j,k+kp)*u(3,i+ip,j,k+kp))/u(0,i+ip,j,k+kp));
	}
	__syncthreads();
	frc2[kp*blockDim.x+ip]=0.0;
	if(((k+kp)<(nz-2))&&((i+ip)<(nx-2))&&(kp<(blockDim.x-1))&&(ip<(blockDim.x-1))){frc2[kp*blockDim.x+ip]+=phi1[kp+(ip*blockDim.x)]+phi1[(kp+1)+(ip*blockDim.x)]+phi1[kp+((ip+1)*blockDim.x)]+phi1[(kp+1)+((ip+1)*blockDim.x)]+phi2[kp+(ip*blockDim.x)]+phi2[(kp+1)+(ip*blockDim.x)]+phi2[kp+((ip+1)*blockDim.x)]+phi2[(kp+1)+((ip+1)*blockDim.x)];}
	int loc_max=blockDim.x*blockDim.y;
	int dist=(loc_max+1)/2;
	i=threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	META_LOOP(while_loop, 1, PROBLEM_SIZE, false);
	while(loc_max>1){
		if((i<dist)&&((i+dist)<loc_max)){frc2[i]+=frc2[i+dist];}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(i==0){frc[blockIdx.y*gridDim.x+blockIdx.x]=frc2[0]*dxi*dzeta;}
	}
}

int main() {
    METRICS_KERNEL_START
    init_constants();

    double *u; cudaMalloc(&u, BUF_5NXZ); cudaMemset(u, 0, BUF_5NXZ);
    double *frc; cudaMalloc(&frc, BUF_NORM); cudaMemset(frc, 0, BUF_NORM);

    printf("[LOG] lu_pintgr_2: NX=%d NY=%d NZ=%d ITERATIONS=%d\n", NX, NY, NZ, ITERATIONS);
    int t = 16;
    size_t smem = (size_t)3*t*t*sizeof(double);
    dim3 tpb2(t, t);
    dim3 grid(NX, NZ);
    lu_kernel<<<grid, tpb2, smem>>>(u, frc, NX, NY, NZ);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(u); cudaFree(frc);
    return 0;
}
