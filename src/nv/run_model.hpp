#include <string>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "buffers.hpp"

constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

namespace gallopwave
{


struct NVObjDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using NVUniquePtr = std::unique_ptr<T, NVObjDeleter>;



class NVLogger: public nvinfer1::ILogger 
{
public:
    void log(Severity severity, char const *msg) override;
};

class NVModel
{
public:
    NVModel(std::string onnxPath, bool isFP16);

    NVModel(std::string prototxtPath, 
            std::string caffemodelPath,
            std::vector<std::string>& outTensorNames,
            bool isFP16);

    NVModel(std::string enginePath);

    ~NVModel(void);

    void outputEngine(std::string dstPath);

    void run(void);

    void runAsync(void);

    void* getDeviceBuffer(const std::string& tensorName) const;

    void* getHostBuffer(const std::string& tensorName) const;

    void copyInputToDevice();

    void copyOutputToHost();

    void copyInputToDeviceAsync();

    void copyOutputToHostAsync();

private:
    NVLogger logger;
    NVUniquePtr<nvinfer1::ICudaEngine> engine;
    NVUniquePtr<nvinfer1::IExecutionContext> context;
    std::vector<std::unique_ptr<ManagedBuffer>> managedBuffers;
    std::vector<void*> deviceBindings;
    cudaStream_t stream;

    void initBuffers(void);

    void* getBuffer(const bool isHost, const std::string& tensorName) const;

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async);
};


}