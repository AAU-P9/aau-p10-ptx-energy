#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>

// IS_CLASS: 1=S, 2=W, 3=A  (default S)
#ifndef IS_CLASS
#define IS_CLASS 1
#endif

#if IS_CLASS == 1
#define TOTAL_KEYS_LOG_2  16
#define MAX_KEY_LOG_2     11
#define NUM_BUCKETS_LOG_2  9
#define CLASS 'S'
#elif IS_CLASS == 2
#define TOTAL_KEYS_LOG_2  20
#define MAX_KEY_LOG_2     16
#define NUM_BUCKETS_LOG_2 10
#define CLASS 'W'
#elif IS_CLASS == 3
#define TOTAL_KEYS_LOG_2  23
#define MAX_KEY_LOG_2     19
#define NUM_BUCKETS_LOG_2 10
#define CLASS 'A'
#endif

#define TOTAL_KEYS        (1 << TOTAL_KEYS_LOG_2)
#define MAX_KEY           (1 << MAX_KEY_LOG_2)
#define NUM_BUCKETS       (1 << NUM_BUCKETS_LOG_2)
#define NUM_KEYS          TOTAL_KEYS
#define SIZE_OF_BUFFERS   NUM_KEYS
#define MAX_ITERATIONS    10
#define TEST_ARRAY_SIZE    5

typedef int INT_TYPE;

#define R23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define R46 (R23*R23)
#define T23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define T46 (T23*T23)

#define USE_BUCKETS

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif
#ifndef TPB
#define TPB 32
#endif

extern __shared__ INT_TYPE extern_share_data[];

// Buffer sizes
#define BUF_KEY_ARRAY  ((size_t)NUM_KEYS      * sizeof(INT_TYPE))
#define BUF_KEY_BUFF1  ((size_t)MAX_KEY        * sizeof(INT_TYPE))
#define BUF_KEY_BUFF2  ((size_t)NUM_KEYS       * sizeof(INT_TYPE))
#define BUF_TEST_ARRAY ((size_t)TEST_ARRAY_SIZE * sizeof(INT_TYPE))
#define BUF_SUM        ((size_t)TPB             * sizeof(INT_TYPE))

__device__ double randlc_device(double* X,
		double* A){
	double T1, T2, T3, T4;
	double A1;
	double A2;
	double X1;
	double X2;
	double Z;
	INT_TYPE j;

	/*
	 * --------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2 and set X = N.
	 * --------------------------------------------------------------------
	 */
	T1 = R23 * *A;
	j  = T1;
	A1 = j;
	A2 = *A - T23 * A1;

	/*
	 * --------------------------------------------------------------------
	 * break X into two parts such that X = 2^23 * X1 + X2, compute
	 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	 * X = 2^23 * Z + A2 * X2  (mod 2^46). 
	 * --------------------------------------------------------------------
	 */
	T1 = R23 * *X;
	j  = T1;
	X1 = j;
	X2 = *X - T23 * X1;
	T1 = A1 * X2 + A2 * X1;

	j  = R23 * T1;
	T2 = j;
	Z = T1 - T23 * T2;
	T3 = T23 * Z + A2 * X2;
	j  = R46 * T3;
	T4 = j;
	*X = T3 - T46 * T4;

	return(R46 * *X);
} 
__device__ double find_my_seed_device(INT_TYPE kn,
		INT_TYPE np,
		long nn,
		double s,
		double a){
	double t1,t2;
	long mq,nq,kk,ik;

	if(kn==0){return s;}

	mq = (nn/4 + np - 1) / np;
	nq = mq * 4 * kn;

	t1 = s;
	t2 = a;
	kk = nq;
	while(kk > 1){
		ik = kk / 2;
		if(2*ik==kk){
			(void)randlc_device(&t2, &t2);
			kk = ik;
		}else{
			(void)randlc_device(&t1, &t2);
			kk = kk - 1;
		}
	}
	(void)randlc_device(&t1, &t2);

	return(t1);
}

__global__ void is_kernel(INT_TYPE* key_array,
		double seed,
		double a,
		INT_TYPE number_of_blocks,
		INT_TYPE amount_of_work){
	double x, s;
	INT_TYPE i, k;

	INT_TYPE k1, k2;
	double an = a;
	INT_TYPE myid, num_procs;
	INT_TYPE mq;

	myid = blockIdx.x*blockDim.x+threadIdx.x;
	num_procs = amount_of_work;

	mq = (NUM_KEYS + num_procs - 1) / num_procs;
	k1 = mq * myid;
	k2 = k1 + mq;
	if(k2 > NUM_KEYS){k2 = NUM_KEYS;}

	s = find_my_seed_device(myid, num_procs, (long)4*NUM_KEYS, seed, an);

	k = MAX_KEY/4;

	for(i=k1; i<k2; i++){
		x = randlc_device(&s, &an);
		x += randlc_device(&s, &an);
		x += randlc_device(&s, &an);
		x += randlc_device(&s, &an);  
		key_array[i] = k*x;
	}
}

int main() {
    METRICS_KERNEL_START

    INT_TYPE *key_array; cudaMalloc(&key_array, BUF_KEY_ARRAY); cudaMemset(key_array, 0, BUF_KEY_ARRAY);

    int tpb = TPB;
    int amount_of_work = tpb * tpb;
    int grid = (amount_of_work + tpb - 1) / tpb;

    printf("[LOG] is_create_seq_kernel: CLASS=%c TOTAL_KEYS=%d MAX_KEY=%d ITERATIONS=%d\n",
           CLASS, TOTAL_KEYS, MAX_KEY, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        is_kernel<<<grid, tpb, 0>>>(key_array, 314159265.0, 1220703125.0, grid, amount_of_work);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  grid);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", tpb);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(key_array);
    return 0;
}
