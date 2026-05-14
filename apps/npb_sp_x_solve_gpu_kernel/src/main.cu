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

__global__ static void sp_kernel(const double* rho_i, 
		const double* us, 
		const double* speed, 
		double* rhs, 
		double* lhs, 
		double* rhstmp, 
		const int nx, 
		const int ny, 
		const int nz){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
#define lhs(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3)))]
#define lhsp(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m+4)))]
#define lhsm(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(j)+ny*((k)+nz*((i)+nx*(m)))]
	int i, j, k, m;
	double rhon[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	/* coalesced */
	j=blockIdx.x*blockDim.x+threadIdx.x+1;
	k=blockIdx.y*blockDim.y+threadIdx.y+1;

	/* uncoalesced */
	/* k=blockIdx.x*blockDim.x+threadIdx.x+1; */
	/* j=blockIdx.y*blockDim.y+threadIdx.y+1; */

	if((k>=nz-1) || (j>=ny-1)){return;}

	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * computes the left hand side for the three x-factors  
	 * ---------------------------------------------------------------------
	 * first fill the lhs for the u-eigenvalue                   
	 * ---------------------------------------------------------------------
	 */
	_lhs[0][0]=lhsp(0,0,j,k)=0.0;
	_lhs[0][1]=lhsp(1,0,j,k)=0.0;
	_lhs[0][2]=lhsp(2,0,j,k)=1.0;
	_lhs[0][3]=lhsp(3,0,j,k)=0.0;
	_lhs[0][4]=lhsp(4,0,j,k)=0.0;
	#pragma unroll
	META_LOOP(i_sweep, 3, 3, true);
	for(i=0; i<3; i++){
		fac1=c3c4*rho_i(i,j,k);
		rhon[i]=max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), dx1);
		cv[i]=us(i,j,k);
	}
	_lhs[1][0]=0.0;
	_lhs[1][1]=-dttx2*cv[0]-dttx1*rhon[0];
	_lhs[1][2]=1.0+c2dttx1*rhon[1];
	_lhs[1][3]=dttx2*cv[2]-dttx1*rhon[2];
	_lhs[1][4]=0.0;
	_lhs[1][2]+=comz5;
	_lhs[1][3]-=comz4;
	_lhs[1][4]+=comz1;
	#pragma unroll
	META_LOOP(m_vars_1, 5, 5, true);
	for(m=0; m<5; m++){lhsp(m,1,j,k)=_lhs[1][m];}
	rhon[0]=rhon[1];
	rhon[1]=rhon[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	#pragma unroll
	META_LOOP(m3_vars, 3, 3, true);
	for(m=0; m<3; m++){
		_rhs[0][m]=rhs(m,0,j,k);
		_rhs[1][m]=rhs(m,1,j,k);
	}
	/*
	 * ---------------------------------------------------------------------
	 * FORWARD ELIMINATION  
	 * ---------------------------------------------------------------------
	 * perform the thomas algorithm; first, FORWARD ELIMINATION     
	 * ---------------------------------------------------------------------
	 */
	META_LOOP(i_sweep_1, 1, NX, false);
	for(i=0; i<nx-2; i++){
		/*
		 * ---------------------------------------------------------------------
		 * first fill the lhs for the u-eigenvalue                   
		 * ---------------------------------------------------------------------
		 */
		if((i+2)==(nx-1)){
			_lhs[2][0]=lhsp(0,i+2,j,k)=0.0;
			_lhs[2][1]=lhsp(1,i+2,j,k)=0.0;
			_lhs[2][2]=lhsp(2,i+2,j,k)=1.0;
			_lhs[2][3]=lhsp(3,i+2,j,k)=0.0;
			_lhs[2][4]=lhsp(4,i+2,j,k)=0.0;
		}else{
			fac1=c3c4*rho_i(i+3,j,k);
			rhon[2]=max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), dx1);
			cv[2]=us(i+3,j,k);
			_lhs[2][0]=0.0;
			_lhs[2][1]=-dttx2*cv[0]-dttx1*rhon[0];
			_lhs[2][2]=1.0+c2dttx1*rhon[1];
			_lhs[2][3]=dttx2*cv[2]-dttx1*rhon[2];
			_lhs[2][4]=0.0;
			/*
			 * ---------------------------------------------------------------------
			 * add fourth order dissipation                             
			 * ---------------------------------------------------------------------
			 */
			if((i+2)==(2)){
				_lhs[2][1]-=comz4;
				_lhs[2][2]+=comz6;
				_lhs[2][3]-=comz4;
				_lhs[2][4]+=comz1;
			}else if((i+2>=3) && (i+2<nx-3)){
				_lhs[2][0]+=comz1;
				_lhs[2][1]-=comz4;
				_lhs[2][2]+=comz6;
				_lhs[2][3]-=comz4;
				_lhs[2][4]+=comz1;
			}else if((i+2)==(nx-3)){
				_lhs[2][0]+=comz1;
				_lhs[2][1]-=comz4;
				_lhs[2][2]+=comz6;
				_lhs[2][3]-=comz4;
			}else if((i+2)==(nx-2)){
				_lhs[2][0]+=comz1;
				_lhs[2][1]-=comz4;
				_lhs[2][2]+=comz5;
			}
			/*
			 * ---------------------------------------------------------------------
			 * store computed lhs for later reuse
			 * ---------------------------------------------------------------------
			 */
			#pragma unroll
			META_LOOP(m_vars_2, 5, 5, true);
			for(m=0;m<5;m++){lhsp(m,i+2,j,k)=_lhs[2][m];}
			rhon[0]=rhon[1];
			rhon[1]=rhon[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		/*
		 * ---------------------------------------------------------------------
		 * load rhs values for current iteration
		 * ---------------------------------------------------------------------
		 */
		#pragma unroll
		META_LOOP(m3_vars_1, 3, 3, true);
		for(m=0;m<3;m++){_rhs[2][m]=rhs(m,i+2,j,k);}
		/*
		 * ---------------------------------------------------------------------
		 * perform current iteration
		 * ---------------------------------------------------------------------
		 */
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		#pragma unroll
		META_LOOP(m3_vars_2, 3, 3, true);
		for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		#pragma unroll
		META_LOOP(m3_vars_3, 3, 3, true);
		for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		#pragma unroll
		META_LOOP(m3_vars_4, 3, 3, true);
		for(m=0;m<3;m++){_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];}
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced 
		 * ---------------------------------------------------------------------
		 */
		lhs(3,i,j,k)=_lhs[0][3];
		lhs(4,i,j,k)=_lhs[0][4];
		#pragma unroll
		META_LOOP(m_vars_3, 5, 5, true);
		for(m=0; m<5; m++){
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
		#pragma unroll
		META_LOOP(m3_vars_5, 3, 3, true);
		for(m=0; m<3; m++){
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * the last two rows in this zone are a bit different,  
	 * since they do not have two more rows available for the
	 * elimination of off-diagonal entries    
	 * ---------------------------------------------------------------------
	 */
	i=nx-2;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	#pragma unroll
	META_LOOP(m3_vars_6, 3, 3, true);
	for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	#pragma unroll
	META_LOOP(m3_vars_7, 3, 3, true);
	for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	fac1=1.0/_lhs[1][2];
	#pragma unroll
	META_LOOP(m3_vars_8, 3, 3, true);
	for(m=0;m<3;m++){_rhs[1][m]*=fac1;}
	lhs(3,nx-2,j,k)=_lhs[0][3];
	lhs(4,nx-2,j,k)=_lhs[0][4];
	/*
	 * ---------------------------------------------------------------------
	 * subsequently, fill the other factors (u+c), (u-c)
	 * ---------------------------------------------------------------------
	 */
	#pragma unroll
	META_LOOP(i_sweep_2, 3, 3, true);
	for(i=0;i<3;i++){cv[i]=speed(i,j,k);}
	#pragma unroll
	META_LOOP(m_vars_4, 5, 5, true);
	for(m=0; m<5; m++){
		_lhsp[0][m]=_lhs[0][m]=lhsp(m,0,j,k);
		_lhsp[1][m]=_lhs[1][m]=lhsp(m,1,j,k);
	}
	_lhsp[1][1]-= dttx2*cv[0];
	_lhsp[1][3]+=dttx2*cv[2];
	_lhs[1][1]+=dttx2*cv[0];
	_lhs[1][3]-=dttx2*cv[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	_rhs[0][3]=rhs(3,0,j,k);
	_rhs[0][4]=rhs(4,0,j,k);
	_rhs[1][3]=rhs(3,1,j,k);
	_rhs[1][4]=rhs(4,1,j,k);
	/*
	 * ---------------------------------------------------------------------
	 * do the u+c and the u-c factors                 
	 * ---------------------------------------------------------------------
	 */
	META_LOOP(i_sweep_3, 1, NX, false);
	for(i=0; i<nx-2; i++){
		/*
		 * first, fill the other factors (u+c), (u-c) 
		 * ---------------------------------------------------------------------
		 */
		#pragma unroll
		META_LOOP(m_vars_5, 5, 5, true);
		for(m=0; m<5; m++){
			_lhsp[2][m]=_lhs[2][m]=lhsp(m,i+2,j,k);
		}
		_rhs[2][3]=rhs(3,i+2,j,k);
		_rhs[2][4]=rhs(4,i+2,j,k);
		if((i+2)<(nx-1)){
			cv[2]=speed(i+3,j,k);
			_lhsp[2][1]-=dttx2*cv[0];
			_lhsp[2][3]+=dttx2*cv[2];
			_lhs[2][1]+=dttx2*cv[0];
			_lhs[2][3]-=dttx2*cv[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		m=3;
		fac1=1.0/_lhsp[0][2];
		_lhsp[0][3]*=fac1;
		_lhsp[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
		_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
		_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
		_lhsp[2][1]-=_lhsp[2][0]*_lhsp[0][3];
		_lhsp[2][2]-=_lhsp[2][0]*_lhsp[0][4];
		_rhs[2][m]-=_lhsp[2][0]*_rhs[0][m];
		m=4;
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced  
		 * ---------------------------------------------------------------------
		 */
		#pragma unroll
		META_LOOP(m_vars_6, 5, 5, true);
		for(m=3; m<5; m++){
			lhsp(m,i,j,k)=_lhsp[0][m];
			lhsm(m,i,j,k)=_lhs[0][m];
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
		#pragma unroll
		META_LOOP(m_vars_7, 5, 5, true);
		for(m=0; m<5; m++){
			_lhsp[0][m]=_lhsp[1][m];
			_lhsp[1][m]=_lhsp[2][m];
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * and again the last two rows separately 
	 * ---------------------------------------------------------------------
	 */
	i=nx-2;
	m=3;
	fac1=1.0/_lhsp[0][2];
	_lhsp[0][3]*=fac1;
	_lhsp[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
	_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
	_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
	m=4;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	_rhs[1][3]/=_lhsp[1][2];
	_rhs[1][4]/=_lhs[1][2];
	/*
	 * ---------------------------------------------------------------------
	 * BACKSUBSTITUTION 
	 * ---------------------------------------------------------------------
	 */
	#pragma unroll
	META_LOOP(m3_vars_9, 3, 3, true);
	for(m=0;m<3;m++){_rhs[0][m]-=lhs(3,nx-2,j,k)*_rhs[1][m];}
	_rhs[0][3]-=_lhsp[0][3]*_rhs[1][3];
	_rhs[0][4]-=_lhs[0][3]*_rhs[1][4];
	#pragma unroll
	META_LOOP(m_vars_8, 5, 5, true);
	for(m=0; m<5; m++){
		_rhs[2][m]=_rhs[1][m];
		_rhs[1][m]=_rhs[0][m];
	}
	META_LOOP(i_sweep_back, 1, PROBLEM_SIZE, false);
	for(i=nx-3; i>=0; i--){
		/*
		 * ---------------------------------------------------------------------
		 * the first three factors
		 * ---------------------------------------------------------------------
		 */
		#pragma unroll
		META_LOOP(m3_vars_10, 3, 3, true);
		for(m=0; m<3; m++){_rhs[0][m]=rtmp(m,i,j,k)-lhs(3,i,j,k)*_rhs[1][m]-lhs(4,i,j,k)*_rhs[2][m];}
		/*
		 * ---------------------------------------------------------------------
		 * and the remaining two
		 * ---------------------------------------------------------------------
		 */
		_rhs[0][3]=rtmp(3,i,j,k)-lhsp(3,i,j,k)*_rhs[1][3]-lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4]=rtmp(4,i,j,k)-lhsm(3,i,j,k)*_rhs[1][4]-lhsm(4,i,j,k)*_rhs[2][4];
		if(i+2<nx-1){
			/*
			 * ---------------------------------------------------------------------
			 * do the block-diagonal inversion          
			 * ---------------------------------------------------------------------
			 */
			double r1=_rhs[2][0];
			double r2=_rhs[2][1];
			double r3=_rhs[2][2];
			double r4=_rhs[2][3];
			double r5=_rhs[2][4];
			double t1=bt*r3;
			double t2=0.5*(r4+r5);
			_rhs[2][0]=-r2;
			_rhs[2][1]=r1;
			_rhs[2][2]=bt*(r4-r5);
			_rhs[2][3]=-t1+t2;
			_rhs[2][4]=t1+t2;
		}
		#pragma unroll
		META_LOOP(m_vars_9, 5, 5, true);
		for(m=0; m<5; m++){
			rhs(m,i+2,j,k)=_rhs[2][m];
			_rhs[2][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[0][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * do the block-diagonal inversion          
	 * ---------------------------------------------------------------------
	 */
	double t1=bt*_rhs[2][2];
	double t2=0.5*(_rhs[2][3]+_rhs[2][4]);
	rhs(0,1,j,k)=-_rhs[2][1];
	rhs(1,1,j,k)=_rhs[2][0];
	rhs(2,1,j,k)=bt*(_rhs[2][3]-_rhs[2][4]);
	rhs(3,1,j,k)=-t1+t2;
	rhs(4,1,j,k)=t1+t2;
	#pragma unroll
	META_LOOP(m_vars_10, 5, 5, true);
	for(m=0;m<5;m++){rhs(m,0,j,k)=_rhs[1][m];}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
	}
}

int main() {
    METRICS_KERNEL_START
    init_constants();

    double *rho_i; cudaMalloc(&rho_i, BUF_NXZ); cudaMemset(rho_i, 0, BUF_NXZ);
    double *us; cudaMalloc(&us, BUF_NXZ); cudaMemset(us, 0, BUF_NXZ);
    double *speed; cudaMalloc(&speed, BUF_NXZ); cudaMemset(speed, 0, BUF_NXZ);
    double *rhs; cudaMalloc(&rhs, BUF_5NXZ); cudaMemset(rhs, 0, BUF_5NXZ);
    double *lhs; cudaMalloc(&lhs, BUF_LHS); cudaMemset(lhs, 0, BUF_LHS);
    double *rhs_buf; cudaMalloc(&rhs_buf, BUF_5NXZ); cudaMemset(rhs_buf, 0, BUF_5NXZ);

    printf("[LOG] sp_x_solve_gpu_kernel: NX=%d NY=%d NZ=%d ITERATIONS=%d\n", NX, NY, NZ, ITERATIONS);
    dim3 grid(1, NZ);
    sp_kernel<<<grid, NY>>>(rho_i, us, speed, rhs, lhs, rhs_buf, NX, NY, NZ);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  1);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(rho_i); cudaFree(us); cudaFree(speed); cudaFree(rhs); cudaFree(lhs); cudaFree(rhs_buf);
    return 0;
}
