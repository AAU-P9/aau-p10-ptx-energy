#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

// PROBLEM_SIZE sets nx=ny=nz; 12=class S, 36=class W, 64=class A
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

// Access macros (match sp.cu)
#define u(m,i,j,k)       u      [(i)+NX*((j)+NY*((k)+NZ*(m)))]
#define forcing(m,i,j,k) forcing[(i)+NX*((j)+NY*((k)+NZ*(m)))]
#define rhs(m,i,j,k)     rhs    [(m)+(i)*5+(j)*5*NX+(k)*5*NX*NY]
#define rho_i(i,j,k)     rho_i  [(i)+(j)*NX+(k)*NX*NY]
#define us(i,j,k)        us     [(i)+(j)*NX+(k)*NX*NY]
#define vs(i,j,k)        vs     [(i)+(j)*NX+(k)*NX*NY]
#define ws(i,j,k)        ws     [(i)+(j)*NX+(k)*NX*NY]
#define square(i,j,k)    square [(i)+(j)*NX+(k)*NX*NY]
#define qs(i,j,k)        qs     [(i)+(j)*NX+(k)*NX*NY]
#define speed(i,j,k)     speed  [(i)+(j)*NX+(k)*NX*NY]

extern __shared__ double extern_share_data[];

// Buffer sizes
#define BUF_5NXZ  ((size_t)5*NX*NY*NZ*sizeof(double))
#define BUF_NXZ   ((size_t)  NX*NY*NZ*sizeof(double))
#define BUF_LHS   ((size_t)9*NX*NY*NZ*sizeof(double))
#define BUF_RMS   ((size_t)5*NX*NY*sizeof(double))

namespace constants_device {
    __constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3,
             dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4,
             dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt,
             dxmax, dymax, dzmax, xxcon1, xxcon2,
             xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
             dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
             yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
             zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1,
             dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1,
             dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2,
             c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
             dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1,
             c2dtty1, c2dttz1, comz1, comz4, comz5, comz6,
             c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16,
             ce[13][5];
}

static void init_constants() {
    double ce[13][5] = {
        {2.0,1.0,2.0,2.0,5.0},{0.0,0.0,2.0,2.0,4.0},{0.0,0.0,0.0,0.0,3.0},
        {4.0,0.0,0.0,0.0,2.0},{5.0,1.0,0.0,0.0,0.1},{3.0,2.0,2.0,2.0,0.4},
        {0.5,3.0,3.0,3.0,0.3},{0.02,0.01,0.04,0.03,0.05},{0.01,0.03,0.03,0.05,0.04},
        {0.03,0.02,0.05,0.04,0.03},{0.5,0.4,0.3,0.2,0.1},{0.4,0.3,0.5,0.1,0.3},
        {0.3,0.5,0.4,0.3,0.2}
    };
    double c1=1.4, c2=0.4, c3=0.1, c4=1.0, c5=1.4;
    double bt=sqrt(0.5), dt=0.015;
    double dnxm1=1.0/(NX-1), dnym1=1.0/(NY-1), dnzm1=1.0/(NZ-1);
    double c1c2=c1*c2, c1c5=c1*c5, c3c4=c3*c4, c1345=c1c5*c3c4;
    double conz1=1.0-c1c5;
    double tx1=1.0/(dnxm1*dnxm1), tx2=1.0/(2.0*dnxm1), tx3=1.0/dnxm1;
    double ty1=1.0/(dnym1*dnym1), ty2=1.0/(2.0*dnym1), ty3=1.0/dnym1;
    double tz1=1.0/(dnzm1*dnzm1), tz2=1.0/(2.0*dnzm1), tz3=1.0/dnzm1;
    double dx1=0.75, dx2=0.75, dx3=0.75, dx4=0.75, dx5=0.75;
    double dy1=0.75, dy2=0.75, dy3=0.75, dy4=0.75, dy5=0.75;
    double dz1=1.0,  dz2=1.0,  dz3=1.0,  dz4=1.0,  dz5=1.0;
    double dxmax=fmax(dx3,dx4), dymax=fmax(dy2,dy4), dzmax=fmax(dz2,dz3);
    double dssp=0.25*fmax(dx1,fmax(dy1,dz1));
    double c4dssp=4.0*dssp, c5dssp=5.0*dssp;
    double dttx1=dt*tx1, dttx2=dt*tx2, dtty1=dt*ty1, dtty2=dt*ty2;
    double dttz1=dt*tz1, dttz2=dt*tz2;
    double c2dttx1=2.0*dttx1, c2dtty1=2.0*dtty1, c2dttz1=2.0*dttz1;
    double dtdssp=dt*dssp;
    double comz1=dtdssp, comz4=4.0*dtdssp, comz5=5.0*dtdssp, comz6=6.0*dtdssp;
    double c3c4tx3=c3c4*tx3, c3c4ty3=c3c4*ty3, c3c4tz3=c3c4*tz3;
    double dx1tx1=dx1*tx1, dx2tx1=dx2*tx1, dx3tx1=dx3*tx1, dx4tx1=dx4*tx1, dx5tx1=dx5*tx1;
    double dy1ty1=dy1*ty1, dy2ty1=dy2*ty1, dy3ty1=dy3*ty1, dy4ty1=dy4*ty1, dy5ty1=dy5*ty1;
    double dz1tz1=dz1*tz1, dz2tz1=dz2*tz1, dz3tz1=dz3*tz1, dz4tz1=dz4*tz1, dz5tz1=dz5*tz1;
    double c2iv=2.5, con43=4.0/3.0, con16=1.0/6.0;
    double xxcon1=c3c4tx3*con43*tx3, xxcon2=c3c4tx3*tx3, xxcon3=c3c4tx3*conz1*tx3;
    double xxcon4=c3c4tx3*con16*tx3, xxcon5=c3c4tx3*c1c5*tx3;
    double yycon1=c3c4ty3*con43*ty3, yycon2=c3c4ty3*ty3, yycon3=c3c4ty3*conz1*ty3;
    double yycon4=c3c4ty3*con16*ty3, yycon5=c3c4ty3*c1c5*ty3;
    double zzcon1=c3c4tz3*con43*tz3, zzcon2=c3c4tz3*tz3, zzcon3=c3c4tz3*conz1*tz3;
    double zzcon4=c3c4tz3*con16*tz3, zzcon5=c3c4tz3*c1c5*tz3;
#define CSYM(name) cudaMemcpyToSymbol(constants_device::name, &name, sizeof(double))
    cudaMemcpyToSymbol(constants_device::ce, ce, 13*5*sizeof(double));
    CSYM(bt); CSYM(dt); CSYM(c1); CSYM(c2); CSYM(c3); CSYM(c4); CSYM(c5);
    CSYM(dnxm1); CSYM(dnym1); CSYM(dnzm1); CSYM(c1c2); CSYM(c1c5); CSYM(c3c4);
    CSYM(c1345); CSYM(conz1); CSYM(tx1); CSYM(tx2); CSYM(tx3); CSYM(ty1); CSYM(ty2);
    CSYM(ty3); CSYM(tz1); CSYM(tz2); CSYM(tz3); CSYM(dx1); CSYM(dx2); CSYM(dx3);
    CSYM(dx4); CSYM(dx5); CSYM(dy1); CSYM(dy2); CSYM(dy3); CSYM(dy4); CSYM(dy5);
    CSYM(dz1); CSYM(dz2); CSYM(dz3); CSYM(dz4); CSYM(dz5); CSYM(dxmax); CSYM(dymax);
    CSYM(dzmax); CSYM(dssp); CSYM(c4dssp); CSYM(c5dssp); CSYM(dttx1); CSYM(dttx2);
    CSYM(dtty1); CSYM(dtty2); CSYM(dttz1); CSYM(dttz2); CSYM(c2dttx1); CSYM(c2dtty1);
    CSYM(c2dttz1); CSYM(dtdssp); CSYM(comz1); CSYM(comz4); CSYM(comz5); CSYM(comz6);
    CSYM(c3c4tx3); CSYM(c3c4ty3); CSYM(c3c4tz3); CSYM(dx1tx1); CSYM(dx2tx1); CSYM(dx3tx1);
    CSYM(dx4tx1); CSYM(dx5tx1); CSYM(dy1ty1); CSYM(dy2ty1); CSYM(dy3ty1); CSYM(dy4ty1);
    CSYM(dy5ty1); CSYM(dz1tz1); CSYM(dz2tz1); CSYM(dz3tz1); CSYM(dz4tz1); CSYM(dz5tz1);
    CSYM(c2iv); CSYM(con43); CSYM(con16); CSYM(xxcon1); CSYM(xxcon2); CSYM(xxcon3);
    CSYM(xxcon4); CSYM(xxcon5); CSYM(yycon1); CSYM(yycon2); CSYM(yycon3); CSYM(yycon4);
    CSYM(yycon5); CSYM(zzcon1); CSYM(zzcon2); CSYM(zzcon3); CSYM(zzcon4); CSYM(zzcon5);
#undef CSYM
}
__device__ static void exact_solution_gpu_device(const double xi,
		const double eta,
		const double zeta,
		double* dtemp){
	using namespace constants_device;
	#pragma unroll
	META_LOOP(m_vars, 5, 5, true);
	for(int m=0; m<5; m++){
		dtemp[m]=ce[0][m]+xi*
			(ce[1][m]+xi*
			 (ce[4][m]+xi*
			  (ce[7][m]+xi*
			   ce[10][m])))+eta*
			(ce[2][m]+eta*
			 (ce[5][m]+eta*
			  (ce[8][m]+eta*
			   ce[11][m])))+zeta*
			(ce[3][m]+zeta*
			 (ce[6][m]+zeta*
			  (ce[9][m]+zeta*
			   ce[12][m])));
	}
}

__global__ static void sp_kernel(double* rms,
		const int nx,
		const int ny,
		const int nz){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	int i, m, maxpos, dist;

	double* buffer = (double*)extern_share_data;

	i = threadIdx.x;

	#pragma unroll
	META_LOOP(m_vars_1, 5, 5, true);
	for(m=0;m<5;m++){buffer[i+(m*blockDim.x)]=0.0;}
	META_LOOP(while_loop, 1, PROBLEM_SIZE, false);
	while(i<nx*ny){
		#pragma unroll
		META_LOOP(m_vars_2, 5, 5, true);
		for(m=0;m<5;m++){buffer[threadIdx.x+(m*blockDim.x)]+=rms[i+nx*ny*m];}
		i+=blockDim.x;
	}
	maxpos=blockDim.x;
	dist=(maxpos+1)/2;
	i=threadIdx.x;
	__syncthreads();
	META_LOOP(while_loop_1, 1, PROBLEM_SIZE, false);
	while(maxpos>1){
		if(i<dist && i+dist<maxpos){
			#pragma unroll
			META_LOOP(m_vars_3, 5, 5, true);
			for(m=0;m<5;m++){buffer[i+(m*blockDim.x)]+=buffer[(i+dist)+(m*blockDim.x)];}
		}
		maxpos=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	m=threadIdx.x;
	if(m<5){rms[m]=sqrt(buffer[0+(m*blockDim.x)]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));}
	}
}

int main() {
    METRICS_KERNEL_START
    init_constants();

    double *rms; cudaMalloc(&rms, BUF_RMS); cudaMemset(rms, 0, BUF_RMS);

    printf("[LOG] sp_error_norm_gpu_kernel_2: NX=%d NY=%d NZ=%d ITERATIONS=%d\n", NX, NY, NZ, ITERATIONS);
    size_t smem = (size_t)TPB*5*sizeof(double);
    sp_kernel<<<1, TPB, smem>>>(rms, NX, NY, NZ);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(rms);
    return 0;
}
