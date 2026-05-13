#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Problem dimensions — defaults match NPB class S
#ifndef NX
#define NX 64
#endif
#ifndef NY
#define NY 64
#endif
#ifndef NZ
#define NZ 64
#endif

#define NTOTAL    ((size_t)NX * NY * NZ)
#define NITER_DEFAULT 6
#define CHECKSUM_TASKS 1024

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif
#ifndef TPB
#define TPB 32
#endif

// dcomplex type and operations (from npb-CPP.hpp)
typedef struct { double real; double imag; } dcomplex;
#define dcomplex_create(r,i)  (dcomplex){r, i}
#define dcomplex_add(a,b)     (dcomplex){(a).real+(b).real, (a).imag+(b).imag}
#define dcomplex_mul2(a,b)    (dcomplex){(a).real*(b), (a).imag*(b)}

#define AP (-4.0*1.0e-6*3.141592653589793238*3.141592653589793238)
#define A  (1220703125.0)

#define R23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define R46 (R23*R23)
#define T23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define T46 (T23*T23)

extern __shared__ double extern_share_data[];

// Buffer sizes
#define BUF_NTOTAL_COMPLEX  (NTOTAL * sizeof(dcomplex))
#define BUF_NTOTAL_DOUBLE   (NTOTAL * sizeof(double))
#define BUF_STARTS          ((size_t)NZ * sizeof(double))
#define BUF_SUMS            ((size_t)(NITER_DEFAULT+1) * sizeof(dcomplex))
// MAXDIM = max(NX,NY,NZ); safe for the three classes we target
#define BUF_U_COMPLEX       ((size_t)(NX > NY ? (NX > NZ ? NX : NZ) : (NY > NZ ? NY : NZ)) * sizeof(dcomplex))

__device__ int ilog2_device(int n){
	int nn, lg;
	if(n==1){
		return 0;
	}
	lg = 1;
	nn = 2;
	while(nn<n){
		nn = nn << 1;
		lg++;
	}
	return lg;
}
__device__ double randlc_device(double* x, 
		double a){
	double t1,t2,t3,t4,a1,a2,x1,x2,z;
	t1 = R23 * a;
	a1 = (int)t1;
	a2 = a - T23 * a1;
	t1 = R23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - T23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(R23 * t1);
	z = t1 - T23 * t2;
	t3 = T23 * z + a2 * x2;
	t4 = (int)(R46 * t3);
	(*x) = t3 - T46 * t4;
	return (R46 * (*x));
}
__device__ void vranlc_device(int n, 
		double* x_seed, 
		double a, 
		double y[]){
	int i;
	double x,t1,t2,t3,t4,a1,a2,x1,x2,z;
	t1 = R23 * a;
	a1 = (int)t1;
	a2 = a - T23 * a1;
	x = *x_seed;
	for(i=0; i<n; i++){
		t1 = R23 * x;
		x1 = (int)t1;
		x2 = x - T23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)(R23 * t1);
		z = t1 - T23 * t2;
		t3 = T23 * z + a2 * x2;
		t4 = (int)(R46 * t3);
		x = t3 - T46 * t4;
		y[i] = R46 * x;
	}
	*x_seed = x;
}
__device__ void ipow46_device(double a, 
		int exponent, 
		double* result){
	double q, r;
	int n, n2;
	/*
	 * --------------------------------------------------------------------
	 * use
	 * a^n = a^(n/2)*a^(n/2) if n even else
	 * a^n = a*a^(n-1)       if n odd
	 * -------------------------------------------------------------------
	 */
	*result = 1;
	if(exponent==0){return;}
	q = a;
	r = 1;
	n = exponent;
	while(n>1){
		n2 = n/2;
		if(n2*2==n){
			randlc_device(&q, q);
			n = n2;
		}else{
			randlc_device(&r, q);
			n = n-1;
		}
	}
	randlc_device(&r, q);
	*result = r;
}
__device__ void cffts3_gpu_fftz2_device(const int is, 
		int l, 
		int m, 
		int n, 
		dcomplex u[], 
		dcomplex x[], 
		dcomplex y[], 
		int index_arg, 
		int size_arg){
	int k,n1,li,lj,lk,ku,i,i11,i12,i21,i22;
	double x11real, x11imag;
	double x21real, x21imag;
	dcomplex u1;
	/*
	 * ---------------------------------------------------------------------
	 * set initial parameters.
	 * ---------------------------------------------------------------------
	 */
	n1 = n / 2;
	lk = 1 << (l - 1);
	li = 1 << (m - l);
	lj = 2 * lk;
	ku = li;
	for(i=0; i<li; i++){
		i11 = i * lk;
		i12 = i11 + n1;
		i21 = i * lj;
		i22 = i21 + lk;
		if(is>=1){
			u1.real = u[ku+i].real;
			u1.imag = u[ku+i].imag;
		}else{
			u1.real = u[ku+i].real;
			u1.imag = -u[ku+i].imag;
		}
		for(k=0; k<lk; k++){
			x11real = x[(i11+k)*size_arg+index_arg].real;
			x11imag = x[(i11+k)*size_arg+index_arg].imag;
			x21real = x[(i12+k)*size_arg+index_arg].real;
			x21imag = x[(i12+k)*size_arg+index_arg].imag;
			y[(i21+k)*size_arg+index_arg].real = x11real + x21real;
			y[(i21+k)*size_arg+index_arg].imag = x11imag + x21imag;
			y[(i22+k)*size_arg+index_arg].real = u1.real * (x11real - x21real) - u1.imag * (x11imag - x21imag);
			y[(i22+k)*size_arg+index_arg].imag = u1.real * (x11imag - x21imag) + u1.imag * (x11real - x21real);
		}
	}
}
__device__ void cffts3_gpu_cfftz_device(const int is, 
		int m, 
		int n, 
		dcomplex x[], 
		dcomplex y[], 
		dcomplex u_device[], 
		int index_arg, 
		int size_arg){
	int j,l;
	/*
	 * ---------------------------------------------------------------------
	 * perform one variant of the Stockham FFT.
	 * ---------------------------------------------------------------------
	 */
	for(l=1; l<=m; l+=2){
		cffts3_gpu_fftz2_device(is, l, m, n, u_device, x, y, index_arg, size_arg);
		if(l==m){break;}
		cffts3_gpu_fftz2_device(is, l + 1, m, n, u_device, y, x, index_arg, size_arg);
	}
	/*
	 * ---------------------------------------------------------------------
	 * copy Y to X.
	 * ---------------------------------------------------------------------
	 */
	if(m%2==1){
		for(j=0; j<n; j++){
			x[j*size_arg+index_arg].real = y[j*size_arg+index_arg].real;
			x[j*size_arg+index_arg].imag = y[j*size_arg+index_arg].imag;
		}
	}
}

__global__ void ft_kernel(dcomplex u0[], 
		double starts[]){    
	int z = blockIdx.x * blockDim.x + threadIdx.x;

	if(z>=NZ){return;}

	double x0 = starts[z];	
	for(int y=0; y<NY; y++){
		vranlc_device(2*NX, &x0, A, (double*)&u0[ 0 + y*NX + z*NX*NY ]);
	}
}

int main() {
    METRICS_KERNEL_START

    dcomplex *u0; cudaMalloc(&u0, BUF_NTOTAL_COMPLEX); cudaMemset(u0, 0, BUF_NTOTAL_COMPLEX);
    double *starts; cudaMalloc(&starts, BUF_STARTS); cudaMemset(starts, 0, BUF_STARTS);

    int grid = (int)(((size_t)NZ + TPB - 1) / TPB);

    printf("[LOG] ft_compute_initial_conditions_kernel: NX=%d NY=%d NZ=%d ITERATIONS=%d\n", NX, NY, NZ, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        ft_kernel<<<grid, TPB, 0>>>(u0, starts);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", (int)grid);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", TPB);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(u0); cudaFree(starts);
    return 0;
}
