#ifdef USE_LLI

typedef enum ParameterType {
    Int,
    Float,
    UnsignedInt,
    SignedInt,
    Int64,
    Int32,
    Int16,
    Int8,
    Int4,
    Unknown
} ParameterType;

typedef struct Parameter {
    char* name;
    ParameterType type;
    size_t size;
    void* value;
} Parameter;

typedef struct KernelParameters {
    dim3 gridDim;
    dim3 blockDim;
    Parameter *parameters;
    unsigned int parameters_length;
} KernelParameters;

extern "C" {

void madsen_function(void **args);

cudaError_t cudaMalloc(void **p, size_t s) { *p = nullptr; return cudaSuccess; }
cudaError_t cudaFree(void *p) { return cudaSuccess; }

cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim,
                                     dim3 blockDim,
                                     size_t sharedMem,
                                     void *stream) {

    printf("gridDim = (%d, %d, %d)\n", 
        gridDim.x, gridDim.y, gridDim.z);
    printf("blockDim = (%d, %d, %d)\n", 
           blockDim.x, blockDim.y, blockDim.z);
    return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
                                       dim3 *blockDim,
                                       size_t *sharedMem,
                                       void *stream) {
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func,
                             dim3 gridDim,
                             dim3 blockDim,
                             void **args,
                             size_t sharedMem,
                             void *stream) {
    /*int m = *(int*)args[3];
    int n = *(int*)args[4];
    int k = *(int*)args[5];

    printf("m: %d\n", m);
    printf("n: %d\n", n);
    printf("k: %d\n", k);

    exit(0);*/

    madsen_function(args);

    return cudaSuccess;
}

}
#endif