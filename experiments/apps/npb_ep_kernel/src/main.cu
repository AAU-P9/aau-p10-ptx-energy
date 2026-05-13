#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

#ifndef M_EP
#define M_EP 24
#endif

#define MK          (16)
#define MM          (M_EP - MK)
#define NN          (1 << MM)
#define NK          (1 << MK)
#define NQ          (10)
#define EP_A        (1220703125.0)
#define EP_S        (271828183.0)
#define RECOMPUTATION (128)

#define R23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define R46 (R23*R23)
#define T23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define T46 (T23*T23)

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif

#ifndef TPB
#define TPB 32
#endif

__device__ double randlc_device(double* x, double a) {
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

__device__ void vranlc_device(int n, double* x_seed, double a, double* y) {
    int i;
    double x,t1,t2,t3,t4,a1,a2,x1,x2,z;
    t1 = R23 * a;
    a1 = (int)t1;
    a2 = a - T23 * a1;
    x = *x_seed;
    for (i = 0; i < n; i++) {
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

__global__ void ep_kernel(double* q_global, double* sx_global, double* sy_global, double an) {
    double x_local[2*RECOMPUTATION];
    double q_local[NQ];
    double sx_local, sy_local;
    double t1, t2, t3, t4, x1, x2, seed;
    int i, ii, ik, kk, l;

    q_local[0]=0.0; q_local[1]=0.0; q_local[2]=0.0; q_local[3]=0.0; q_local[4]=0.0;
    q_local[5]=0.0; q_local[6]=0.0; q_local[7]=0.0; q_local[8]=0.0; q_local[9]=0.0;
    sx_local = 0.0;
    sy_local = 0.0;

    kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (kk >= NN) { return; }

    t1 = EP_S;
    t2 = an;

    for (i = 1; i <= 100; i++) {
        ik = kk / 2;
        if ((2*ik) != kk) { t3 = randlc_device(&t1, t2); }
        if (ik == 0) { break; }
        t3 = randlc_device(&t2, t2);
        kk = ik;
    }

    seed = t1;
    for (ii = 0; ii < NK; ii = ii + RECOMPUTATION) {
        vranlc_device(2*RECOMPUTATION, &seed, EP_A, x_local);
        for (i = 0; i < RECOMPUTATION; i++) {
            x1 = 2.0*x_local[2*i]   - 1.0;
            x2 = 2.0*x_local[2*i+1] - 1.0;
            t1 = x1*x1 + x2*x2;
            if (t1 <= 1.0) {
                t2 = sqrt(-2.0*log(t1)/t1);
                t3 = (x1*t2);
                t4 = (x2*t2);
                l = (int)fmax(fabs(t3), fabs(t4));
                q_local[l] += 1.0;
                sx_local += t3;
                sy_local += t4;
            }
        }
    }

    atomicAdd(q_global + blockIdx.x*NQ+0, q_local[0]);
    atomicAdd(q_global + blockIdx.x*NQ+1, q_local[1]);
    atomicAdd(q_global + blockIdx.x*NQ+2, q_local[2]);
    atomicAdd(q_global + blockIdx.x*NQ+3, q_local[3]);
    atomicAdd(q_global + blockIdx.x*NQ+4, q_local[4]);
    atomicAdd(q_global + blockIdx.x*NQ+5, q_local[5]);
    atomicAdd(q_global + blockIdx.x*NQ+6, q_local[6]);
    atomicAdd(q_global + blockIdx.x*NQ+7, q_local[7]);
    atomicAdd(q_global + blockIdx.x*NQ+8, q_local[8]);
    atomicAdd(q_global + blockIdx.x*NQ+9, q_local[9]);
    atomicAdd(sx_global + blockIdx.x, sx_local);
    atomicAdd(sy_global + blockIdx.x, sy_local);
}

int main() {
    METRICS_KERNEL_START

    int tpb = TPB;
    int blocks = (NN + tpb - 1) / tpb;

    size_t sz_q  = (size_t)blocks * NQ * sizeof(double);
    size_t sz_sx = (size_t)blocks * sizeof(double);
    size_t sz_sy = (size_t)blocks * sizeof(double);

    double *q, *sx, *sy;
    cudaMalloc(&q,  sz_q);  cudaMemset(q,  0, sz_q);
    cudaMalloc(&sx, sz_sx); cudaMemset(sx, 0, sz_sx);
    cudaMalloc(&sy, sz_sy); cudaMemset(sy, 0, sz_sy);

    const double an = EP_A;

    printf("[LOG] ep_kernel: M_EP=%d NN=%d blocks=%d tpb=%d ITERATIONS=%d\n",
           M_EP, NN, blocks, tpb, ITERATIONS);
    for (int it = 0; it < ITERATIONS; it++) {
        ep_kernel<<<blocks, tpb>>>(q, sx, sy, an);
    }
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  blocks);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", tpb);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);
    EXPORT_N("M_EP",       M_EP);

    METRICS_KERNEL_END

    cudaFree(q);
    cudaFree(sx);
    cudaFree(sy);
    return 0;
}
