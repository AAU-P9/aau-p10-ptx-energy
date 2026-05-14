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
	META_LOOP(while_loop, 1, PROBLEM_SIZE, false);
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
		INT_TYPE* partial_verify_vals,
		INT_TYPE* test_index_array,
		INT_TYPE iteration,
		INT_TYPE number_of_blocks,
		INT_TYPE amount_of_work){
	META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
	for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	key_array[iteration] = iteration;
	key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
	/*
	 * --------------------------------------------------------------------
	 * determine where the partial verify test keys are, 
	 * --------------------------------------------------------------------
	 * load into top of array bucket_size  
	 * --------------------------------------------------------------------
	 */
#pragma unroll
	#pragma unroll
	META_LOOP(i_sweep, TEST_ARRAY_SIZE, TEST_ARRAY_SIZE, true);
	for(INT_TYPE i=0; i<TEST_ARRAY_SIZE; i++){
		partial_verify_vals[i] = key_array[test_index_array[i]];
	}
	}
}

int main() {
    METRICS_KERNEL_START

    INT_TYPE *key_array; cudaMalloc(&key_array, BUF_KEY_ARRAY); cudaMemset(key_array, 0, BUF_KEY_ARRAY);
    INT_TYPE *partial_verify_vals; cudaMalloc(&partial_verify_vals, BUF_TEST_ARRAY); cudaMemset(partial_verify_vals, 0, BUF_TEST_ARRAY);
    INT_TYPE *test_index_array; cudaMalloc(&test_index_array, BUF_TEST_ARRAY); cudaMemset(test_index_array, 0, BUF_TEST_ARRAY);

    int tpb = 1;
    int grid = 1;

    printf("[LOG] is_rank_kernel_1: CLASS=%c TOTAL_KEYS=%d MAX_KEY=%d ITERATIONS=%d\n",
           CLASS, TOTAL_KEYS, MAX_KEY, ITERATIONS);
    is_kernel<<<grid, tpb, 0>>>(key_array, partial_verify_vals, test_index_array, 0, 1, 1);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  grid);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", tpb);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(key_array); cudaFree(partial_verify_vals); cudaFree(test_index_array);
    return 0;
}
