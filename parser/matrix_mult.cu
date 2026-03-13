#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define M 4
#define N 3
#define K 5

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k, int s, float z) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < m && col < k) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

int main() {
    int m = M, n = N, k = K;

    int *h_a = (int*)malloc(m * n * sizeof(int));
    int *h_b = (int*)malloc(n * k * sizeof(int));
    int *h_c = (int*)malloc(m * k * sizeof(int));

    srand(240901);
    for (int i = 0; i < m * n; i++) h_a[i] = rand() % 10; // 0-9
    for (int i = 0; i < n * k; i++) h_b[i] = rand() % 10; // 0-9

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * n * sizeof(int));
    cudaMalloc(&d_b, n * k * sizeof(int));
    cudaMalloc(&d_c, m * k * sizeof(int));

    cudaMemcpy(d_a, h_a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * k * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int s = 21049;
    float z = 499843.323;

    gpu_matrix_mult<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, m, n, k, s, z);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, m * k * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) printf("%d ", h_a[i * n + j]);
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) printf("%d ", h_b[i * k + j]);
        printf("\n");
    }

    printf("\nMatrix C = A * B:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) printf("%d ", h_c[i * k + j]);
        printf("\n");
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}