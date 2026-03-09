#ifdef USE_LLI

#include <iostream>
#include <string>
#include <iomanip>

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

void PrintKernelParametersJSON(const KernelParameters &kp)
{
    std::cout << "{\n";
    std::cout << "  \"gridDim\": { \"x\": " << kp.gridDim.x << ", \"y\": " << kp.gridDim.y << ", \"z\": " << kp.gridDim.z << " },\n";
    std::cout << "  \"blockDim\": { \"x\": " << kp.blockDim.x << ", \"y\": " << kp.blockDim.y << ", \"z\": " << kp.blockDim.z << " },\n";
    std::cout << "  \"parameters_length\": " << kp.parameters_length << ",\n";
    std::cout << "  \"parameters\": [\n";

    for (unsigned int i = 0; i < kp.parameters_length; ++i)
    {
        const Parameter &p = kp.parameters[i];
        std::cout << "    {\n";
        std::cout << "      \"name\": \"" << p.name << "\",\n";
        std::cout << "      \"type\": \"" << ParameterTypeToString(p.type) << "\",\n";
        std::cout << "      \"size\": " << p.size << ",\n";
        std::cout << "      \"value\": ";

        switch (p.type)
        {
        case Int:
            std::cout << *((int *)p.value);
            break;
        case UnsignedInt:
            std::cout << *((unsigned int *)p.value);
            break;
        case Float:
            std::cout << std::fixed << std::setprecision(6) << *((float *)p.value);
            break;
        case Int64:
            std::cout << *((long long *)p.value);
            break;
        case Int32:
            std::cout << *((int32_t *)p.value);
            break;
        case Int16:
            std::cout << *((int16_t *)p.value);
            break;
        case Int8:
            std::cout << (int)(*((int8_t *)p.value));
            break;
        default:
            std::cout << "null";
            break;
        }

        std::cout << "\n    }";
        if (i + 1 < kp.parameters_length)
            std::cout << ",";
        std::cout << "\n";
    }

    std::cout << "  ]\n";
    std::cout << "}\n";
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
}
#endif