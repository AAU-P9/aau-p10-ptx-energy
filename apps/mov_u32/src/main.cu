#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>

using namespace std;

// Number of samples to collect for CPU/GPU timestamp offsets
#define OFFSET_SAMPLES 10

// Cupti buffer management parameters
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align_size) (((uintptr_t) (buffer) + (align_size) - 1) & ~((align_size) - 1))

// Global variables to store kernel timing information
static uint64_t kernel_duration = 0;
static uint64_t start_time = 0;
static uint64_t end_time = 0;

void bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    *buffer = (uint8_t *)ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *size = BUF_SIZE;
    *maxNumRecords = 0;
}

void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    if (validSize > 0) {
        CUpti_Activity *record = NULL;
        
        while (CUPTI_SUCCESS == cuptiActivityGetNextRecord(buffer, validSize, &record)) {
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                // start timestamp is at offset 2*8 = 16 bytes
                // end timestamp is at offset 3*8 = 24 bytes
                uint64_t *start_ptr = (uint64_t *)((uint8_t*)record + 16);
                uint64_t *end_ptr = (uint64_t *)((uint8_t*)record + 24);

                start_time = *start_ptr;
                end_time = *end_ptr;
                kernel_duration = *end_ptr - *start_ptr;
            }
        }
    }
    free(buffer);
}

__global__ void ptx_kernel(int *out, int iterations)
{
    int tid = threadIdx.x;
    int tmp = tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < iterations; ++i)
    {
        asm volatile (
            "mov.u32 %0, %0;\n\t"  // move tmp to tmp (self-move)
            : "+r"(tmp)             // %0 is a register mapped to tmp
        );
    }
}

int main()
{
    int h[4] = {0}; 
    int *d;
    int iterations = 100000000; // 100 million iterations

    // Initialize CUPTI profiling
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);

    cudaMalloc(&d, 4*sizeof(int));
    cudaMemcpy(d, h, 4*sizeof(int), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", iterations);

    // Get CPU/GPU offsets
    for (int i = 0; i < OFFSET_SAMPLES; ++i) {
        uint64_t cuptiTimestamp = 0;
        auto cpuInitial = chrono::high_resolution_clock::now();
        cuptiGetTimestamp(&cuptiTimestamp);
        auto cpuFinal = chrono::high_resolution_clock::now();

        printf("[OFFSET] CUPTI Timestamp: %lu, CPU Timestamp #1: %lu, CPU Timestamp #2: %lu\n", 
            cuptiTimestamp, 
            chrono::duration_cast<chrono::nanoseconds>(cpuInitial.time_since_epoch()).count(),
            chrono::duration_cast<chrono::nanoseconds>(cpuFinal.time_since_epoch()).count()
        );

    }

    // Run kernel
    ptx_kernel<<<1,4>>>(d, iterations);
    cudaDeviceSynchronize();

    // Possibly read back results (not necessary for timing, but included for completeness)
    // cudaMemcpy(h, d, 4*sizeof(int), cudaMemcpyDeviceToHost);

    // Flush all activity buffers
    cuptiActivityFlushAll(0);

    if (kernel_duration > 0) {
        printf("[KERNEL] Start Time: %lu\n", start_time);
        printf("[KERNEL] End Time: %lu\n", end_time);
        printf("[KERNEL] Duration: %lu\n", kernel_duration);
    }

    // Clean up
    cudaFree(d);
    
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
}
