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
#define ITERATIONS 0
#define VIRTUALWINDOW	100
#define _N 96
#define _M 96
#define blockSideFIL 1024
#define blockSideCOL 1
#define gridSideFIL	(((_N) + (blockSideFIL) - 1) / (blockSideFIL))
#define gridSideCOL	(((_M) + (blockSideCOL) - 1) / (blockSideCOL))
#define _blockSize ((blockSideFIL) * (blockSideCOL))

/* Kernel definition and implementation*/
__global__ void KERNEL_LAUNCH_BOUNDS(1024, 1)
matrix(int *A){

  META_BEGIN_KERNEL(matrix);
  META_PARAM_PTR(0, A, ELEM_I32, output_matrix, ALIGN(ALIGN_CACHELINE) ACCESS_WRITEONLY);
  META_TILE(DIM_X, 1);
  META_TILE(DIM_Y, 1024);
  META_LAUNCH(1, 1024, 1, "96 1 1");
  META_LAYOUT(A, LAYOUT_LINEAR, "_N*_M");
  META_ASSUME("blockDim.x == 1 && blockDim.y == 1024");

	/* To get the global block ID in the grid*/
	int blockGLobalId=	blockIdx.y*gridDim.x+ blockIdx.x;

	/* To get the global thread ID in the grid independently of the block shape*/
	int finalResultPos=	(blockGLobalId*_blockSize)+ (threadIdx.y*blockDim.x+threadIdx.x);

	/* Stotring the final result*/
	A[finalResultPos]= KERNEL_NUMBER;

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
	for(i=0;i<_N;i++){

		for(j=0;j<_M;j++){

			/* Initialization*/
			matrix[i*_N+j]= INITIALIZATION;

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
  initializeCUPTI();

  /* Matrix creation */
  int *A = (int *)malloc(sizeof(int) * (_N * _M));
  if (A == NULL) {
    printf("Error malloc\n");
    exit(-1);
  }

  /* Allocate device memory for the matrix `dA` used by the kernel. */
  size_t bytes = (size_t)_N * (size_t)_M * sizeof(int);
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
  cudaMemcpy(dA, A, sizeof(int) * (_N * _M), cudaMemcpyHostToDevice);

  collectTimestampOffsets();
  benchmark();
  flushCUPTIBuffers();
  printKernelTiming();

  free(A);
  cudaFree(dA);

  return 0;
}
