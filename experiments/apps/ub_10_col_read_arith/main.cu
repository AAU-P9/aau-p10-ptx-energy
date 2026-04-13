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
#define ITERATIONS  2000
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
  META_LAYOUT(A, LAYOUT_ROW_MAJOR, "N x M");
  META_ASSUME("blockDim.x == 1 && blockDim.y == 1024");
  META_ASSUME("read access is column-major: A[i*M + row]");
  META_ASSUME("ITERATIONS == 2000");

	/* Index and local variables*/
	int i;
	int sum_aux= 0;

	/* Local variables to access to output matrix*/
	int result_point_row	= blockIdx.y*blockDim.y + threadIdx.y;
	int result_point_col	= blockIdx.x*blockDim.x + threadIdx.x;

	/* Accessing to all elements of the matrix*/
  META_LOOP(col_reduce, 96, 96, false);
	for (i=0; i < M; i++){
		sum_aux+= A[i*M + result_point_row];
	}// loop i

	/* Overload loop*/
  META_LOOP(overload_loop, 2000, 2000, false);
	for(i=0; i< ITERATIONS; i++){
		sum_aux+= i%threadIdx.x;
	}//for


	/* Stotring the final result*/
	A[result_point_row*M + result_point_col]= sum_aux;

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
  benchmark();
  METRICS_KERNEL_END

  free(A);
  cudaFree(dA);

  return 0;
}
