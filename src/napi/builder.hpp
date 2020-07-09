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


namespace gallopwave
{
class OperandType
{
public:
    uint32_t index;
    std::vector<uint32_t> dimensions;
    uint32_t sizeBytes;
    std::string name;
    float *data;                            // TODO: Support only FP32 now
    int fd;                                 // for shared memory
    ANeuralNetworksMemory *nnMemPtr;        // mapped memory to shared memory
};

class ModelBuilder
{
public:
    ModelBuilder(void);
    ~ModelBuilder(void);
    void getSdkVersion (void);
    void getDevices (void);
    void addTensor (std::string name, std::vector<uint32_t> dims,
                    const void *srcbuffer = nullptr, float scale = 0.0f, int32_t zeroPoint = 0);
    void conv2d (std::string name, const std::string &input, const std::string &weight,
                const std::string &bias, const int32_t padLeft, const int32_t padRight,
                const int32_t padTop, const int32_t padBottom, const int32_t strideX,
                const int32_t strideY, const FuseCode fusecode, const bool isNCHW,
                const int32_t dilationX, const int32_t dilationY, const std::string &output);
    void setInputOps (std::string name, float* dataptr);
    void setOutputOps (std::string name);
    void compile(int32_t deviceIndex = -1);
    void execute(void);
    std::vector<float *> getOutput(void);
private:
    uint32_t opIdx;
    std::map<std::string, uint32_t> operandIdxes;
    std::map<std::string, std::vector<uint32_t>> shapeIdxes;
    std::vector<OperandType> inputOps;
    std::vector<OperandType> outputOps;

    ANeuralNetworksModel *model;
    ANeuralNetworksCompilation *compilation;
    ANeuralNetworksExecution *execution;
    ANeuralNetworksEvent *event;
    std::vector<ANeuralNetworksDevice *> devices;
    
};

}