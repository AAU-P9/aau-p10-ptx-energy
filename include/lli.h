#ifdef USE_LLI

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cupti.h>

typedef enum ParameterType
{
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

typedef struct Parameter
{
    char *name;
    ParameterType type;
    size_t size;
    void *value;
} Parameter;

typedef struct KernelParameters
{
    dim3 gridDim;
    dim3 blockDim;
    Parameter *parameters;
    unsigned int parameters_length;
} KernelParameters;

static KernelParameters g_kernelParams;

std::string ParameterTypeToString(ParameterType type)
{
    switch (type)
    {
    case Int:
        return "Int";
    case Float:
        return "Float";
    case UnsignedInt:
        return "UnsignedInt";
    case SignedInt:
        return "SignedInt";
    case Int64:
        return "Int64";
    case Int32:
        return "Int32";
    case Int16:
        return "Int16";
    case Int8:
        return "Int8";
    case Int4:
        return "Int4";
    default:
        return "Unknown";
    }
}

void PrintKernelParametersJSON(const KernelParameters &kp, const std::string &filename = "kernel_params.json")
{
    std::ofstream outfile(filename);

    outfile << "{\n";
    outfile << "  \"gridDim\": { \"x\": " << kp.gridDim.x << ", \"y\": " << kp.gridDim.y << ", \"z\": " << kp.gridDim.z << " },\n";
    outfile << "  \"blockDim\": { \"x\": " << kp.blockDim.x << ", \"y\": " << kp.blockDim.y << ", \"z\": " << kp.blockDim.z << " },\n";
    outfile << "  \"parameters_length\": " << kp.parameters_length << ",\n";
    outfile << "  \"parameters\": [\n";

    for (unsigned int i = 0; i < kp.parameters_length; ++i)
    {
        const Parameter &p = kp.parameters[i];
        outfile << "    {\n";
        outfile << "      \"name\": \"" << p.name << "\",\n";
        outfile << "      \"type\": \"" << ParameterTypeToString(p.type) << "\",\n";
        outfile << "      \"size\": " << p.size << ",\n";
        outfile << "      \"value\": ";

        switch (p.type)
        {
        case Int:
            outfile << *((int *)p.value);
            break;
        case UnsignedInt:
            outfile << *((unsigned int *)p.value);
            break;
        case Float:
            outfile << std::fixed << std::setprecision(6) << *((float *)p.value);
            break;
        case Int64:
            outfile << *((long long *)p.value);
            break;
        case Int32:
            outfile << *((int32_t *)p.value);
            break;
        case Int16:
            outfile << *((int16_t *)p.value);
            break;
        case Int8:
            outfile << (int)(*((int8_t *)p.value));
            break;
        default:
            outfile << "null";
            break;
        }

        outfile << "\n    }";
        if (i + 1 < kp.parameters_length)
            outfile << ",";
        outfile << "\n";
    }

    outfile << "  ]\n";
    outfile << "}\n";
    outfile.close();
}

extern "C"
{

    void madsen_function(void **args, KernelParameters *kp);

    cudaError_t cudaMalloc(void **p, size_t s)
    {
        *p = nullptr;
        return cudaSuccess;
    }
    cudaError_t cudaFree(void *p) { return cudaSuccess; }

    cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
    {
        return cudaSuccess;
    }

    cudaError_t cudaDeviceSynchronize()
    {
        return cudaSuccess;
    }

    unsigned __cudaPushCallConfiguration(dim3 gridDim,
                                         dim3 blockDim,
                                         size_t sharedMem,
                                         void *stream)
    {

        g_kernelParams.gridDim = gridDim;
        g_kernelParams.blockDim = blockDim;
        return 0;
    }

    cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
                                           dim3 *blockDim,
                                           size_t *sharedMem,
                                           void *stream)
    {
        return cudaSuccess;
    }

    cudaError_t cudaLaunchKernel(const void *func,
                                 dim3 gridDim,
                                 dim3 blockDim,
                                 void **args,
                                 size_t sharedMem,
                                 void *stream)
    {
        madsen_function(args, &g_kernelParams);

        return cudaSuccess;
    }

    CUptiResult cuptiActivityFlushAll(uint32_t flag)
    {
        return CUPTI_SUCCESS;
    }

    CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind)
    {
        return CUPTI_SUCCESS;
    }

    CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind)
    {
        return CUPTI_SUCCESS;
    }

    CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer,
                                           size_t validBufferSizeBytes,
                                           CUpti_Activity **record)
    {
        *record = nullptr;
        return CUPTI_ERROR_MAX_LIMIT_REACHED;
    }

    CUptiResult cuptiGetTimestamp(uint64_t *timestamp)
    {
        if (timestamp)
            *timestamp = 0;
        return CUPTI_SUCCESS;
    }

    CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested,
                                               CUpti_BuffersCallbackCompleteFunc funcBufferCompleted)
    {
        return CUPTI_SUCCESS;
    }
}
#endif