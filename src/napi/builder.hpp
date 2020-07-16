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
    void *data;
    int fd;                                     // for shared memory
    ANeuralNetworksMemory *nnMemPtr;            // mapped memory to shared memory
};

class ModelBuilder
{
public:
    ModelBuilder(void);
    ~ModelBuilder(void);
    void getSdkVersion (void);
    void getDevices (void);
    void addTensor (std::string name, std::vector<uint32_t> dims, int32_t opType,
                    const void *srcbuffer = nullptr, float scale = 0.1f, int32_t zeroPoint = 0);

    void conv2d (const std::string& name, const std::string& input, const std::string& weight,
                const std::string& bias, int32_t opType, int32_t padLeft, int32_t padRight,
                int32_t padTop, int32_t padBottom, int32_t strideX, int32_t strideY,
                bool isDepthWise, FuseCode fusecode, const std::string& output,
                float scale = 0.1f, int32_t zeroPoint = 0);

    void eltwiseAdd (const std::string& name, const std::string& input1, const std::string& input2,
                    FuseCode fusecode, const std::string& output, int32_t opType,
                    float scale = 0.1f, int32_t zeroPoint = 0);

    void setInputOps (std::string name, void* dataptr, int32_t opType);
    void setOutputOps (std::string name, int32_t opType);
    void compile(int32_t deviceIndex = -1);
    void execute(void);
    std::vector<void *> getOutput(void);

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

    size_t getElementSize(int32_t opType);
};

}