/*
 *  Benchmark: UBench
 *  Author: Trasgo Research Group
 *  Source: https://trasgo.infor.uva.es/ubench/
 */

#include "ptx_meta.h"
#include <stdio.h>
#include <stdlib.h>
#include "cupti_timing.h"

#define currentGpu     0
#define INITIALIZATION 0
#define KERNEL_NUMBER 99
#ifndef ITERATIONS
#define ITERATIONS 30000000
#endif
#define VIRTUALWINDOW	100
#define N 96
#define M 96
#define blockSideFIL 1024
#define blockSideCOL 1
#define gridSideFIL	(((N) + (blockSideFIL) - 1) / (blockSideFIL))
#define gridSideCOL	(((M) + (blockSideCOL) - 1) / (blockSideCOL))
#define blockSize blockSideFIL*blockSideCOL

/* Kernel definition and implementation*/
__global__ void KERNEL_LAUNCH_BOUNDS(1024, 1)
matrix(int *A){

  META_BEGIN_KERNEL(matrix);
  META_PARAM_PTR(0, A, ELEM_I32, matrix, ALIGN(ALIGN_CACHELINE) ACCESS_READWRITE);
  META_TILE(DIM_X, 1);
  META_TILE(DIM_Y, 1024);
  META_LAUNCH(1, 1024, 1, "96 1 1");
  META_LAYOUT(A, LAYOUT_LINEAR, "N*M");
  META_ASSUME("blockDim.x == 1 && blockDim.y == 1024");
  META_ASSUME("blockSize == 1024");
  META_ASSUME("reads entire block region then writes back to same position");

	int blockGLobalId=	blockIdx.y*gridDim.x+ blockIdx.x;
	int finalResultPos=	(blockGLobalId*blockSize)+ (threadIdx.y*blockDim.x+threadIdx.x);
	if (finalResultPos >= N * M) return;

  META_LOOP(iter_loop, ITERATIONS, ITERATIONS, false);
  for (int _iter = 0; _iter < ITERATIONS; _iter++) {
	  int i;
	  int localIndex;
	  int localValue=0;
    META_LOOP(block_reduce, 1024, 1024, false);
	  for (i=0; i<blockSize;i++){
		  localIndex= blockGLobalId*blockSize + i;
		  localValue+= A[localIndex];
	  }
	  A[finalResultPos]= localValue;
  }

  META_END_KERNEL(matrix);
}//__global__


dim3 dimBlock(blockSideCOL,blockSideFIL);
dim3 dimGrid(gridSideCOL,gridSideFIL);

/* Device matrix declaration*/
int *dA;

/* Matrix initialization*/
void inicialization(int *matrix){

	/* Index variables*/
	int i,j;

	/* Accesing to matrices elements*/
	for(i=0;i<N;i++){

		for(j=0;j<M;j++){

			/* Initialization*/
			matrix[i*N+j]= INITIALIZATION;

		}//for j
	}//for i

}//inicializationMatrix


void benchmark()
{
  for (int i = 0; i < 10000; i++) {
    matrix<<<dimGrid,dimBlock>>>(dA);
    cudaDeviceSynchronize();
  }
}

int main(int argc, char *argv[])
{
  METRICS_KERNEL_START

  /* Matrix creation */
  int *A = (int *)malloc(sizeof(int) * (N * M));
  if (A == NULL) {
    printf("Error malloc\n");
    exit(-1);
  }

  /* Allocate device memory for the matrix `dA` used by the kernel. */
  size_t bytes = (size_t)N * (size_t)M * sizeof(int);
  cudaError_t err = cudaMalloc((void **)&dA, bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for dA (bytes=%zu): %s\n", bytes,
            cudaGetErrorString(err));
    return -1;
  }
  err = cudaMemset(dA, 0, bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemset failed for dA: %s\n", cudaGetErrorString(err));
    cudaFree(dA);
    return -1;
  }

  /* Matrix initialization*/
  inicialization(A);

  /* Copying the matrix elements to device memory*/
  cudaMemcpy(dA, A, sizeof(int) * (N * M), cudaMemcpyHostToDevice);
  matrix<<<dimGrid,dimBlock>>>(dA);
  cudaDeviceSynchronize();

  EXPORT_N("gridDim_x", dimGrid.x);
  EXPORT_N("gridDim_y", dimGrid.y);
  EXPORT_N("gridDim_z", dimGrid.z);
  EXPORT_N("blockDim_x", dimBlock.x);
  EXPORT_N("blockDim_y", dimBlock.y);
  EXPORT_N("blockDim_z", dimBlock.z);
  METRICS_KERNEL_END

  free(A);
  cudaFree(dA);

  return 0;
}
