#pragma once

#include <vector>
#include <string>
#include <map>

#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/system_properties.h>
#include <sys/mman.h>


#define CHECK_NNAPI_ERROR(status)                                                                   \
        if (status != ANEURALNETWORKS_NO_ERROR)                                                     \
        {                                                                                           \
            SLOG_ERROR << status << ", line: " <<__LINE__ << std::endl;                             \
            exit(1);                                                                                \
        }


enum EltwiseCode
{
    ELTWISE_MULTIPLY = 0,
    ELTWISE_ADDITION = 1
};

namespace gallopwave
{

class TensorStruct
{
public:
    uint32_t index;
    std::vector<uint32_t> dimensions;
    uint32_t sizeBytes;
    std::string name;
    void *data;
    int fd;                                     // for shared memory
    ANeuralNetworksMemory *nnMemPtr;            // mapped memory to shared memory
};

class ModelBuilder
{
public:
    ModelBuilder(void);
    ~ModelBuilder(void);
    void getSdkVersion(void);
    void getDevices(void);
    void addTensor(std::string name, std::vector<uint32_t> dims, OperandCode operandcode,
                    const void *srcbuffer = nullptr, float scale = 0.001f, int32_t zeroPoint = 0);

    void conv2d(const std::string& name, const std::string& input, const std::string& weight,
                const std::string& bias, OperandCode operandcode, int32_t padLeft, int32_t padRight,
                int32_t padTop, int32_t padBottom, int32_t strideX, int32_t strideY,
                bool isDepthWise, FuseCode fusecode, const std::string& output,
                float scale = 0.001f, int32_t zeroPoint = 0);

    void eltwise(const std::string& name, const std::string& input1, const std::string& input2,
                FuseCode fusecode, const std::string& output, OperandCode operandcode, EltwiseCode eltwisecode,
                float scale = 0.001f, int32_t zeroPoint = 0);
    
    void maxpool(const std::string& name, const std::string& input, OperandCode operandcode,
                int32_t padLeft, int32_t padRight, int32_t padTop, int32_t padBottom,
                int32_t strideX, int32_t strideY, int32_t kernelW, int32_t kernelH,
                FuseCode fusecode, const std::string& output, float scale = 0.001f, int32_t zeroPoint = 0);
    
    void reduce(const std::string& name, const std::string& input, OperandCode operandcode,
                const std::string& output, float scale = 0.001f, int32_t zeroPoint = 0);

    void setInputTensors (std::string name, void* dataptr, OperandCode operandcode);
    void setOutputTensors (std::string name, OperandCode operandcode);
    void compile(int32_t deviceIndex = -1);
    void execute(void);
    std::vector<void *> getOutput(void);

private:
    uint32_t opIdx;
    std::map<std::string, uint32_t> operandIdxes;
    std::map<std::string, std::vector<uint32_t>> shapeIdxes;
    std::vector<TensorStruct> inputTensors;
    std::vector<TensorStruct> outputTensors;

    ANeuralNetworksModel *model;
    ANeuralNetworksCompilation *compilation;
    ANeuralNetworksExecution *execution;
    ANeuralNetworksEvent *event;
    std::vector<ANeuralNetworksDevice *> devices;

    size_t getElementSize(OperandCode operandcode);
};

}