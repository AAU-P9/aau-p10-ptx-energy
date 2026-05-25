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

__global__ static void lu_kernel(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	int i, j, k, m, nthreads;
	double q, u31;

	double* flux = (double*)extern_share_data;
	double* rtmp = (double*)flux+(blockDim.x*5);
	double* u21j = (double*)rtmp+(blockDim.x*5);
	double* u31j = (double*)u21j+(blockDim.x);
	double* u41j = (double*)u31j+(blockDim.x);
	double* u51j = (double*)u41j+(blockDim.x);

	double utmp[5];

	k=blockIdx.x+1;
	i=blockIdx.y+1;
	j=threadIdx.x;

	using namespace constants_device;
	META_LOOP(while_loop, 1, PROBLEM_SIZE, false);
	while(j<ny){ 
		nthreads=ny-(j-threadIdx.x);
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * eta-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=rtmp[threadIdx.x*5+2];
		u31=rtmp[threadIdx.x*5+2]/rtmp[threadIdx.x*5+0];
		q=0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x+(1*blockDim.x)]=rtmp[threadIdx.x*5+1]*u31;
		flux[threadIdx.x+(2*blockDim.x)]=rtmp[threadIdx.x*5+2]*u31+C2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(3*blockDim.x)]=rtmp[threadIdx.x*5+3]*u31;
		flux[threadIdx.x+(4*blockDim.x)]=(C1*rtmp[threadIdx.x*5+4]-C2*q)*u31;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(j<(ny-1))){for(m=0;m<5;m++){utmp[m]=frct(m,i,j,k)-ty2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u31=1.0/rtmp[threadIdx.x*5+0];
		u21j[threadIdx.x]=u31*rtmp[threadIdx.x*5+1];
		u31j[threadIdx.x]=u31*rtmp[threadIdx.x*5+2];
		u41j[threadIdx.x]=u31*rtmp[threadIdx.x*5+3];
		u51j[threadIdx.x]=u31*rtmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=ty3*(u21j[threadIdx.x]-u21j[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=(4.0/3.0)*ty3*(u31j[threadIdx.x]-u31j[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=ty3*(u41j[threadIdx.x]-u41j[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*ty3*((u21j[threadIdx.x]*u21j[threadIdx.x]+u31j[threadIdx.x]*u31j[threadIdx.x]+u41j[threadIdx.x]*u41j[threadIdx.x])-(u21j[threadIdx.x-1]*u21j[threadIdx.x-1]+u31j[threadIdx.x-1]*u31j[threadIdx.x-1]+u41j[threadIdx.x-1]*u41j[threadIdx.x-1]))+(1.0/6.0)*ty3*(u31j[threadIdx.x]*u31j[threadIdx.x]-u31j[threadIdx.x-1]*u31j[threadIdx.x-1])+C1*C5*ty3*(u51j[threadIdx.x]-u51j[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(j<(ny-1))){
			utmp[0]+=dy1*ty1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dy2*ty1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dy3*ty1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dy4*ty1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dy5*ty1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(j==1){for(m=0;m<5;m++){frct(m,i,1,k)=utmp[m]-dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,3,k));}}
			if(j==2){for(m=0;m<5;m++){frct(m,i,2,k)=utmp[m]-dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,4,k));}}
			if((j>=3)&&(j<(ny-3))){for(m=0;m<5;m++){frct(m,i,j,k)=utmp[m]-dssp*(rsd(m,i,j-2,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j+2,k));}}
			if(j==(ny-3)){for(m=0;m<5;m++){frct(m,i,ny-3,k)=utmp[m]-dssp*(rsd(m,i,ny-5,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);}}
			if(j==(ny-2)){for(m=0;m<5;m++){frct(m,i,ny-2,k)=utmp[m]-dssp*(rsd(m,i,ny-4,k)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);}}
		}
		j += blockDim.x-2;
	}
	}
}

int main() {
    METRICS_KERNEL_START
    init_constants();

    double *u; cudaMalloc(&u, BUF_5NXZ); cudaMemset(u, 0, BUF_5NXZ);
    double *rsd; cudaMalloc(&rsd, BUF_5NXZ); cudaMemset(rsd, 0, BUF_5NXZ);

    printf("[LOG] lu_erhs_3: NX=%d NY=%d NZ=%d ITERATIONS=%d\n", NX, NY, NZ, ITERATIONS);
    int tpb = (NY < TPB) ? NY : TPB;
    size_t smem = (size_t)(2*tpb*5 + 4*tpb)*sizeof(double);
    dim3 grid(NZ-2, NX-2);
    lu_kernel<<<grid, tpb, smem>>>(u, rsd, NX, NY, NZ);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  grid.x);
    EXPORT_N("gridDim_y",  grid.y);
    EXPORT_N("gridDim_z",  grid.z);
    EXPORT_N("blockDim_x", tpb);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(u); cudaFree(rsd);
    return 0;
}
