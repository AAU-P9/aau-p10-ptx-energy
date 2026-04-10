#ifndef CUPTI_TIMING_H
#define CUPTI_TIMING_H

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define JIM_IMPLEMENTATION
#include "jim.h"

// Number of samples to collect for CPU/GPU timestamp offsets
#define OFFSET_SAMPLES 100

// Cupti buffer management parameters
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align_size) (((uintptr_t) (buffer) + (align_size) - 1) & ~((align_size) - 1))

#define JSON_START jim_object_begin(&json);
#define JSON_END jim_object_end(&json); printJson();

template<typename T> inline void jim_auto(Jim *jim, T x) { jim_integer(jim, (long long)x); }
template<> inline void jim_auto(Jim *jim, double x)      { jim_float(jim, x, 6); }
template<> inline void jim_auto(Jim *jim, float x)       { jim_float(jim, (double)x, 6); }
template<> inline void jim_auto(Jim *jim, const char *x) { jim_string(jim, x); }
template<> inline void jim_auto(Jim *jim, char *x)       { jim_string(jim, x); }

#define EXPORT(X)    jim_member_key(&json, #X); jim_auto(&json, X);
#define EXPORT_N(N, X) jim_member_key(&json, N); jim_auto(&json, X);


// Global variables to store kernel timing information
static uint64_t kernel_duration = 0;
static uint64_t start_time = 0;
static uint64_t end_time = 0;

static Jim json = {.pp = 4};

// CUPTI buffer callback functions
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

// Initialize CUPTI profiling
inline void initializeCUPTI() {
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
}

// Disable CUPTI profiling
inline void disableCUPTI() {
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
}

// Collect CPU/GPU timestamp offsets
inline void collectTimestampOffsets() {

    jim_member_key(&json, "timestamp_offsets");
    jim_array_begin(&json);
    for (int i = 0; i < OFFSET_SAMPLES; ++i) {
        uint64_t cuptiTimestamp = 0;
        struct timespec cpuInitial, cpuFinal;
        clock_gettime(CLOCK_MONOTONIC, &cpuInitial);
        cuptiGetTimestamp(&cuptiTimestamp);
        clock_gettime(CLOCK_MONOTONIC, &cpuFinal);

        uint64_t cpuInitialNs = (uint64_t)cpuInitial.tv_sec * 1000000000ULL + (uint64_t)cpuInitial.tv_nsec;
        uint64_t cpuFinalNs = (uint64_t)cpuFinal.tv_sec * 1000000000ULL + (uint64_t)cpuFinal.tv_nsec;

        jim_object_begin(&json);
        {
            jim_member_key(&json, "cupti_timestamp");
            jim_integer(&json, cuptiTimestamp);
            jim_member_key(&json, "cpu_initial_timestamp");
            jim_integer(&json, cpuInitialNs);
            jim_member_key(&json, "cpu_final_timestamp");
            jim_integer(&json, cpuFinalNs);
        }
        jim_object_end(&json);

        // Sleep for a short time to avoid overwhelming the system
        struct timespec sleepTime = {0, 1000000 * 10}; // 10 ms
        nanosleep(&sleepTime, NULL);
    }
    jim_array_end(&json);
}

// Print kernel timing results
inline void printKernelTiming() {

    if (kernel_duration > 0) {
        jim_member_key(&json, "start_timestamp");
        jim_integer(&json, start_time);
        jim_member_key(&json, "end_timestamp");
        jim_integer(&json, end_time);
        jim_member_key(&json, "duration");
        jim_integer(&json, kernel_duration);
    }
}

inline void printJson() {
    std::ofstream outFile("output.json");
    outFile.write(json.sink, json.sink_count);
    //fwrite(json.sink, json.sink_count, 1, stdout);

}

// Flush CUPTI activity buffers
inline void flushCUPTIBuffers() {
    cuptiActivityFlushAll(0);
}

#define METRICS_KERNEL_START JSON_START initializeCUPTI(); collectTimestampOffsets();

#define METRICS_KERNEL_END flushCUPTIBuffers();printKernelTiming(); disableCUPTI(); JSON_END;

#endif // CUPTI_TIMING_H