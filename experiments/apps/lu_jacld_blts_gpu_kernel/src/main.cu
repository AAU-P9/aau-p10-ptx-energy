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

__global__ static void lu_kernel(const int plane,
		const int klower,
		const int jlower,
		const double* u,
		const double* rho_i,
		const double* qs,
		double* v,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double tmp1, tmp2, tmp3, tmat[5*5], tv[5];
	double r43, c1345, c34;

	k=klower+blockIdx.x+1;
	j=jlower+threadIdx.x+1;

	i=plane-k-j+3;

	if((j>(ny-2))||(i>(nx-2))||(i<1)){return;}

	r43=4.0/3.0;
	c1345=C1*C3*C4*C5;
	c34=C3*C4;
	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * form the first block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j,k-1);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]= -dt*tz1*dz1;
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=-dt*tz2;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*tz2*(-(u(1,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(-c34*tmp2*u(1,i,j,k-1));
	tmat[1+5*1]=-dt*tz2*(u(3,i,j,k-1)*tmp1)-dt*tz1*c34*tmp1-dt*tz1*dz2;
	tmat[1+5*2]=0.0;
	tmat[1+5*3]=-dt*tz2*(u(1,i,j,k-1)*tmp1);
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=-dt*tz2*(-(u(2,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(-c34*tmp2*u(2,i,j,k-1));
	tmat[2+5*1]=0.0;
	tmat[2+5*2]=-dt*tz2*(u(3,i,j,k-1)*tmp1)-dt*tz1*(c34*tmp1)-dt*tz1*dz3;
	tmat[2+5*3]=-dt*tz2*(u(2,i,j,k-1)*tmp1);
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=-dt*tz2*(-(u(3,i,j,k-1)*tmp1)*(u(3,i,j,k-1)*tmp1)+C2*qs(i,j,k-1)*tmp1)-dt*tz1*(-r43*c34*tmp2*u(3,i,j,k-1));
	tmat[3+5*1]=-dt*tz2*(-C2*(u(1,i,j,k-1)*tmp1));
	tmat[3+5*2]=-dt*tz2*(-C2*(u(2,i,j,k-1)*tmp1));
	tmat[3+5*3]=-dt*tz2*(2.0-C2)*(u(3,i,j,k-1)*tmp1)-dt*tz1*(r43*c34*tmp1)-dt*tz1*dz4;
	tmat[3+5*4]=-dt*tz2*C2;
	tmat[4+5*0]=-dt*tz2*((C2*2.0*qs(i,j,k-1)-C1*u(4,i,j,k-1))*u(3,i,j,k-1)*tmp2)-dt*tz1*(-(c34-c1345)*tmp3*(u(1,i,j,k-1)*u(1,i,j,k-1))-(c34-c1345)*tmp3*(u(2,i,j,k-1)*u(2,i,j,k-1))-(r43*c34-c1345)*tmp3*(u(3,i,j,k-1)*u(3,i,j,k-1))-c1345*tmp2*u(4,i,j,k-1));
	tmat[4+5*1]=-dt*tz2*(-C2*(u(1,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(c34-c1345)*tmp2*u(1,i,j,k-1);
	tmat[4+5*2]=-dt*tz2*(-C2*(u(2,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(c34-c1345)*tmp2*u(2,i,j,k-1);
	tmat[4+5*3]=-dt*tz2*(C1*(u(4,i,j,k-1)*tmp1)-C2*(qs(i,j,k-1)*tmp1+u(3,i,j,k-1)*u(3,i,j,k-1)*tmp2))-dt*tz1*(r43*c34-c1345)*tmp2*u(3,i,j,k-1);
	tmat[4+5*4]=-dt*tz2*(C1*(u(3,i,j,k-1)*tmp1))-dt*tz1*c1345*tmp1-dt*tz1*dz5;
	for(m=0;m<5;m++){tv[m]=v(m,i,j,k)-omega*(tmat[m+5*0]*v(0,i,j,k-1)+tmat[m+5*1]*v(1,i,j,k-1)+tmat[m+5*2]*v(2,i,j,k-1)+tmat[m+5*3]*v(3,i,j,k-1)+tmat[m+5*4]*v(4,i,j,k-1));}
	/*
	 * ---------------------------------------------------------------------
	 * form the second block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j-1,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*ty1*dy1;
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=-dt*ty2;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*ty2*(-(u(1,i,j-1,k)*u(2,i,j-1,k))*tmp2)-dt*ty1*(-c34*tmp2*u(1,i,j-1,k));
	tmat[1+5*1]=-dt*ty2*(u(2,i,j-1,k)*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy2;
	tmat[1+5*2]=-dt*ty2*(u(1,i,j-1,k)*tmp1);
	tmat[1+5*3]=0.0;
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=-dt*ty2*(-(u(2,i,j-1,k)*tmp1)*(u(2,i,j-1,k)*tmp1)+C2*(qs(i,j-1,k)*tmp1))-dt*ty1*(-r43*c34*tmp2*u(2,i,j-1,k));
	tmat[2+5*1]=-dt*ty2*(-C2*(u(1,i,j-1,k)*tmp1));
	tmat[2+5*2]=-dt*ty2*((2.0-C2)*(u(2,i,j-1,k)*tmp1))-dt*ty1*(r43*c34*tmp1)-dt*ty1*dy3;
	tmat[2+5*3]=-dt*ty2*(-C2*(u(3,i,j-1,k)*tmp1));
	tmat[2+5*4]=-dt*ty2*C2;
	tmat[3+5*0]=-dt*ty2*(-(u(2,i,j-1,k)*u(3,i,j-1,k))*tmp2)-dt*ty1*(-c34*tmp2*u(3,i,j-1,k));
	tmat[3+5*1]=0.0;
	tmat[3+5*2]=-dt*ty2*(u(3,i,j-1,k)*tmp1);
	tmat[3+5*3]=-dt*ty2*(u(2,i,j-1,k)*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy4;
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*ty2*((C2*2.0*qs(i,j-1,k)-C1*u(4,i,j-1,k))*(u(2,i,j-1,k)*tmp2))-dt*ty1*(-(c34-c1345)*tmp3*(u(1,i,j-1,k)*u(1,i,j-1,k))-(r43*c34-c1345)*tmp3*(u(2,i,j-1,k)*u(2,i,j-1,k))-(c34-c1345)*tmp3*(u(3,i,j-1,k)*u(3,i,j-1,k))-c1345*tmp2*u(4,i,j-1,k));
	tmat[4+5*1]=-dt*ty2*(-C2*(u(1,i,j-1,k)*u(2,i,j-1,k))*tmp2)-dt*ty1*(c34-c1345)*tmp2*u(1,i,j-1,k);
	tmat[4+5*2]=-dt*ty2*(C1*(u(4,i,j-1,k)*tmp1)-C2*(qs(i,j-1,k)*tmp1+u(2,i,j-1,k)*u(2,i,j-1,k)*tmp2))-dt*ty1*(r43*c34-c1345)*tmp2*u(2,i,j-1,k);
	tmat[4+5*3]=-dt*ty2*(-C2*(u(2,i,j-1,k)*u(3,i,j-1,k))*tmp2) - dt*ty1*(c34-c1345)*tmp2*u(3,i,j-1,k);
	tmat[4+5*4]=-dt*ty2*(C1*(u(2,i,j-1,k)*tmp1))-dt*ty1*c1345*tmp1-dt*ty1*dy5;
	for(m=0;m<5;m++){tv[m]=tv[m]-omega*(tmat[m+5*0]*v(0,i,j-1,k)+tmat[m+5*1]*v(1,i,j-1,k)+tmat[m+5*2]*v(2,i,j-1,k)+tmat[m+5*3]*v(3,i,j-1,k)+tmat[m+5*4]*v(4,i,j-1,k));}
	/*
	 * ---------------------------------------------------------------------
	 * form the third block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i-1,j,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*tx1*dx1;
	tmat[0+5*1]=-dt*tx2;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*tx2*(-(u(1,i-1,j,k)*tmp1)*(u(1,i-1,j,k)*tmp1)+C2*qs(i-1,j,k)*tmp1)-dt*tx1*(-r43*c34*tmp2*u(1,i-1,j,k));
	tmat[1+5*1]=-dt*tx2*((2.0-C2)*(u(1,i-1,j,k)*tmp1))-dt*tx1*(r43*c34*tmp1)-dt*tx1*dx2;
	tmat[1+5*2]=-dt*tx2*(-C2*(u(2,i-1,j,k)*tmp1));
	tmat[1+5*3]=-dt*tx2*(-C2*(u(3,i-1,j,k)*tmp1));
	tmat[1+5*4]=-dt*tx2*C2;
	tmat[2+5*0]=-dt*tx2*(-(u(1,i-1,j,k)*u(2,i-1,j,k))*tmp2)-dt*tx1*(-c34*tmp2*u(2,i-1,j,k));
	tmat[2+5*1]=-dt*tx2*(u(2,i-1,j,k)*tmp1);
	tmat[2+5*2]=-dt*tx2*(u(1,i-1,j,k)*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx3;
	tmat[2+5*3]=0.0;
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=-dt*tx2*(-(u(1,i-1,j,k)*u(3,i-1,j,k))*tmp2)-dt*tx1*(-c34*tmp2*u(3,i-1,j,k));
	tmat[3+5*1]=-dt*tx2*(u(3,i-1,j,k)*tmp1);
	tmat[3+5*2]=0.0;
	tmat[3+5*3]=-dt*tx2*(u(1,i-1,j,k)*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx4;
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*tx2*((C2*2.0*qs(i-1,j,k)-C1*u(4,i-1,j,k))*u(1,i-1,j,k)*tmp2)-dt*tx1*(-(r43*c34-c1345)*tmp3*(u(1,i-1,j,k)*u(1,i-1,j,k))-(c34-c1345)*tmp3*(u(2,i-1,j,k)*u(2,i-1,j,k))-(c34-c1345)*tmp3*(u(3,i-1,j,k)*u(3,i-1,j,k))-c1345*tmp2*u(4,i-1,j,k));
	tmat[4+5*1]=-dt*tx2*(C1*(u(4,i-1,j,k)*tmp1)-C2*(u(1,i-1,j,k)*u(1,i-1,j,k)*tmp2+qs(i-1,j,k)*tmp1))-dt*tx1*(r43*c34-c1345)*tmp2*u(1,i-1,j,k);
	tmat[4+5*2]=-dt*tx2*(-C2*(u(2,i-1,j,k)*u(1,i-1,j,k))*tmp2)-dt*tx1*(c34-c1345)*tmp2*u(2,i-1,j,k);
	tmat[4+5*3]=-dt*tx2*(-C2*(u(3,i-1,j,k)*u(1,i-1,j,k))*tmp2)-dt*tx1*(c34-c1345)*tmp2*u(3,i-1,j,k);
	tmat[4+5*4]=-dt*tx2*(C1*(u(1,i-1,j,k)*tmp1))-dt*tx1*c1345*tmp1-dt*tx1*dx5;
	for(m=0;m<5;m++){tv[m]=tv[m]-omega*(tmat[m+0*5]*v(0,i-1,j,k)+tmat[m+5*1]*v(1,i-1,j,k)+tmat[m+5*2]*v(2,i-1,j,k)+tmat[m+5*3]*v(3,i-1,j,k)+tmat[m+5*4]*v(4,i-1,j,k));}
	/*
	 * ---------------------------------------------------------------------
	 * form the block diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1);
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*2.0*(tx1*r43+ty1+tz1)*c34*tmp2*u(1,i,j,k);
	tmat[1+5*1]=1.0+dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1) + dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2);
	tmat[1+5*2]=0.0;
	tmat[1+5*3]=0.0;
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=-dt*2.0*(tx1+ty1*r43+tz1)*c34*tmp2*u(2,i,j,k);
	tmat[2+5*1]=0.0;
	tmat[2+5*2]=1.0+dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1)+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3);
	tmat[2+5*3]=0.0;
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=-dt*2.0*(tx1+ty1+tz1*r43)*c34*tmp2*u(3,i,j,k);
	tmat[3+5*1]=0.0;
	tmat[3+5*2]=0.0;
	tmat[3+5*3]=1.0+dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43)+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4);
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u(1,i,j,k)*u(1,i,j,k))+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u(2,i,j,k)*u(2,i,j,k))+(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u(3,i,j,k)*u(3,i,j,k)))*tmp3+(tx1+ty1+tz1)*c1345*tmp2*u(4,i,j,k));
	tmat[4+5*1]=dt*2.0*tmp2*u(1,i,j,k)*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345));
	tmat[4+5*2]=dt*2.0*tmp2*u(2,i,j,k)*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345));
	tmat[4+5*3]=dt*2.0*tmp2*u(3,i,j,k)*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345));
	tmat[4+5*4]=1.0+dt*2.0*(tx1+ty1+tz1)*c1345*tmp1+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5);
	/*
	 * ---------------------------------------------------------------------
	 * diagonal block inversion
	 * --------------------------------------------------------------------- 
	 * forward elimination
	 * ---------------------------------------------------------------------
	 */
	tmp1=1.0/tmat[0+0*5];
	tmp2=tmp1*tmat[1+0*5];
	tmat[1+1*5]-=tmp2*tmat[0+1*5];
	tmat[1+2*5]-=tmp2*tmat[0+2*5];
	tmat[1+3*5]-=tmp2*tmat[0+3*5];
	tmat[1+4*5]-=tmp2*tmat[0+4*5];
	tv[1]-=tmp2*tv[0];
	tmp2=tmp1*tmat[2+0*5];
	tmat[2+1*5]-=tmp2*tmat[0+1*5];
	tmat[2+2*5]-=tmp2*tmat[0+2*5];
	tmat[2+3*5]-=tmp2*tmat[0+3*5];
	tmat[2+4*5]-=tmp2*tmat[0+4*5];
	tv[2]-=tmp2*tv[0];
	tmp2=tmp1*tmat[3+0*5];
	tmat[3+1*5]-=tmp2*tmat[0+1*5];
	tmat[3+2*5]-=tmp2*tmat[0+2*5];
	tmat[3+3*5]-=tmp2*tmat[0+3*5];
	tmat[3+4*5]-=tmp2*tmat[0+4*5];
	tv[3]-=tmp2*tv[0];
	tmp2=tmp1*tmat[4+0*5];
	tmat[4+1*5]-=tmp2*tmat[0+1*5];
	tmat[4+2*5]-=tmp2*tmat[0+2*5];
	tmat[4+3*5]-=tmp2*tmat[0+3*5];
	tmat[4+4*5]-=tmp2*tmat[0+4*5];
	tv[4]-=tmp2*tv[0];
	tmp1=1.0/tmat[1+1*5];
	tmp2=tmp1*tmat[2+1*5];
	tmat[2+2*5]-=tmp2*tmat[1+2*5];
	tmat[2+3*5]-=tmp2*tmat[1+3*5];
	tmat[2+4*5]-=tmp2*tmat[1+4*5];
	tv[2]-=tmp2*tv[1];
	tmp2=tmp1*tmat[3+1*5];
	tmat[3+2*5]-=tmp2*tmat[1+2*5];
	tmat[3+3*5]-=tmp2*tmat[1+3*5];
	tmat[3+4*5]-=tmp2*tmat[1+4*5];
	tv[3]-=tmp2*tv[1];
	tmp2=tmp1*tmat[4+1*5];
	tmat[4+2*5]-=tmp2*tmat[1+2*5];
	tmat[4+3*5]-=tmp2*tmat[1+3*5];
	tmat[4+4*5]-=tmp2*tmat[1+4*5];
	tv[4]-=tmp2*tv[1];
	tmp1=1.0/tmat[2+2*5];
	tmp2=tmp1*tmat[3+2*5];
	tmat[3+3*5]-=tmp2*tmat[2+3*5];
	tmat[3+4*5]-=tmp2*tmat[2+4*5];
	tv[3]-=tmp2*tv[2];
	tmp2=tmp1*tmat[4+2*5];
	tmat[4+3*5]-=tmp2*tmat[2+3*5];
	tmat[4+4*5]-=tmp2*tmat[2+4*5];
	tv[4]-=tmp2*tv[2];
	tmp1=1.0/tmat[3+3*5];
	tmp2=tmp1*tmat[4+3*5];
	tmat[4+4*5]-=tmp2*tmat[3+4*5];
	tv[4]-=tmp2*tv[3];
	/*
	 * ---------------------------------------------------------------------
	 * back substitution
	 * ---------------------------------------------------------------------
	 */
	v(4,i,j,k)=tv[4]/tmat[4+4*5];
	tv[3]=tv[3]-tmat[3+4*5]*v(4,i,j,k);
	v(3,i,j,k)=tv[3]/tmat[3+3*5];
	tv[2]=tv[2]-tmat[2+3*5]*v(3,i,j,k)-tmat[2+4*5]*v(4,i,j,k);
	v(2,i,j,k)=tv[2]/tmat[2+2*5];
	tv[1]=tv[1]-tmat[1+2*5]*v(2,i,j,k)-tmat[1+3*5]*v(3,i,j,k)-tmat[1+4*5]*v(4,i,j,k);
	v(1,i,j,k)=tv[1]/tmat[1+1*5];
	tv[0]=tv[0]-tmat[0+1*5]*v(1,i,j,k)-tmat[0+2*5]*v(2,i,j,k)-tmat[0+3*5]*v(3,i,j,k)-tmat[0+4*5]*v(4,i,j,k);
	v(0,i,j,k)=tv[0]/tmat[0+0*5];
}

int main() {
    METRICS_KERNEL_START
    init_constants();

    double *u; cudaMalloc(&u, BUF_5NXZ); cudaMemset(u, 0, BUF_5NXZ);
    double *rho_i; cudaMalloc(&rho_i, BUF_NXZ); cudaMemset(rho_i, 0, BUF_NXZ);
    double *qs; cudaMalloc(&qs, BUF_NXZ); cudaMemset(qs, 0, BUF_NXZ);
    double *rsd; cudaMalloc(&rsd, BUF_5NXZ); cudaMemset(rsd, 0, BUF_5NXZ);

    printf("[LOG] lu_jacld_blts_gpu_kernel: NX=%d NY=%d NZ=%d ITERATIONS=%d\n", NX, NY, NZ, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        int plane = 5, klower = 0, jlower = 0;
        lu_kernel<<<1, TPB>>>(plane, klower, jlower, u, rho_i, qs, rsd, NX, NY, NZ);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(u); cudaFree(rho_i); cudaFree(qs); cudaFree(rsd);
    return 0;
}
