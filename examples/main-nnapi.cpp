#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>

#include <unistd.h>

#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/system_properties.h>
#include <sys/mman.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"




static auto LOG = spdlog::stdout_color_mt("MAIN");

#define CHECK_NNAPI_ERROR(status)                                                                   \
        if (status != ANEURALNETWORKS_NO_ERROR)                                                     \
        {                                                                                           \
            SLOG_ERROR << status << ", line: " <<__LINE__ << std::endl;                             \
            exit(1);                                                                                \
        }

void addTensor (ANeuralNetworksModel *model,
                std::map<std::string, uint32_t>& operandIdxes,
                std::map<std::string, std::vector<uint32_t>>& shapeIdxes,
                std::string name,
                uint32_t index,
                std::vector<uint32_t> dims,
                const void *srcbuffer = nullptr,
                float scale = 0.0f,
                int32_t zeroPoint = 0
                )
{
    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_FLOAT32; // support only FP32 now
    operandType.dimensionCount = static_cast<uint32_t>(dims.size());
    operandType.dimensions = dims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name] = index;
    shapeIdxes[name] = dims;

    if (srcbuffer != nullptr)
    {
        size_t elementSize = 4; // support only FP32 now
        const size_t bytes = elementSize * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, srcbuffer, bytes) );
    }
}


void conv2d (ANeuralNetworksModel *model,
            std::map<std::string, uint32_t>& operandIdxes,
            std::map<std::string, std::vector<uint32_t>>& shapeIdxes,
            std::string name,
            uint32_t& index,
            const std::string &input,
            const std::string &weight,
            const std::string &bias,
            const int32_t padLeft,
            const int32_t padRight,
            const int32_t padTop,
            const int32_t padBottom,
            const int32_t strideX,
            const int32_t strideY,
            const FuseCode fusecode,
            const bool isNCHW,
            const int32_t dilationX,
            const int32_t dilationY,
            const std::string &output
            )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    const auto weightIdx = operandIdxes.at(weight);
    const auto biasIdx = operandIdxes.at(bias);
    parameterIdxes.push_back(inputIdx);
    parameterIdxes.push_back(weightIdx);
    parameterIdxes.push_back(biasIdx);




    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = nullptr;
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType));
    operandIdxes[name + "_padLeft"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &padLeft, sizeof(padLeft)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padRight"] = index;
    CHECK_NNAPI_ERROR ( ANeuralNetworksModel_setOperandValue(model, index, &padRight, sizeof(padRight)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padTop"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &padTop, sizeof(padTop)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padBottom"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &padBottom, sizeof(padBottom)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideX"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &strideX, sizeof(strideX)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideY"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &strideY, sizeof(strideY)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_activation"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &fusecode, sizeof(fusecode)) );
    parameterIdxes.push_back(index);
    ++index;

    operandType.type = ANEURALNETWORKS_BOOL;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_isNCHW"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &isNCHW, sizeof(isNCHW)) );
    parameterIdxes.push_back(index);
    ++index;

    operandType.type = ANEURALNETWORKS_INT32;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_dilationX"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &dilationX, sizeof(dilationX)) );
    parameterIdxes.push_back(index);
    ++index;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_dilationY"] = index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, index, &dilationY, sizeof(dilationY)) );
    parameterIdxes.push_back(index);
    ++index;



    const auto inDims = shapeIdxes.at(input);
    const auto wDims = shapeIdxes.at(weight);
    const uint32_t outN = inDims[0];
    const uint32_t outH = (inDims[1] - ((wDims[1] - 1) * dilationY + 1) + padTop + padBottom) / strideY + 1;
    const uint32_t outW = (inDims[2] - ((wDims[2] - 1) * dilationX + 1) + padLeft + padRight) / strideX + 1;
    const uint32_t outC = wDims[0];
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = ANEURALNETWORKS_TENSOR_FLOAT32; // support only FP32 now
    operandType.dimensionCount = static_cast<uint32_t>(inDims.size());
    operandType.dimensions = outDims.data();
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = index;
    shapeIdxes[output] = {outN, outH, outW, outC};

    outIdxes.push_back(index);
    ++index;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
}



void getAndroidSdkVersion()
{
    int32_t result = 0;
    const char* sdkProp = "ro.build.version.sdk";
    char sdkVersion[PROP_VALUE_MAX];
    int length = __system_property_get(sdkProp, sdkVersion);
    if (length != 0) 
    {
        for (int i = 0; i < length; ++i)
        {
            int digit = sdkVersion[i] - '0';
            if (digit < 0 || digit > 9)
            {
                SLOG_INFO << "Non-numeric SDK version, assume it's higher than expected" << std::endl;
            }

            result = result * 10 + digit;
        }
        SLOG_INFO << "Android SDK version: " << result << std::endl;
    }
    else
    {
        SLOG_INFO << "Failed parsing Android SDK version" << std::endl;
    }
}


void getDevices()
{
    uint32_t deviceCount;
    CHECK_NNAPI_ERROR( ANeuralNetworks_getDeviceCount(&deviceCount) );
    // SLOG_INFO << "deviceCount: " << deviceCount << std::endl;
    for (int i = 0; i < deviceCount; ++i)
    {
        ANeuralNetworksDevice *nnDevice;
        CHECK_NNAPI_ERROR( ANeuralNetworks_getDevice(i, &nnDevice) );
        const char *nnDeviceName;
        CHECK_NNAPI_ERROR( ANeuralNetworksDevice_getName(nnDevice, &nnDeviceName) );
        int64_t featureLevel;
        CHECK_NNAPI_ERROR( ANeuralNetworksDevice_getFeatureLevel(nnDevice, &featureLevel) );
        int32_t type;
        CHECK_NNAPI_ERROR( ANeuralNetworksDevice_getType(nnDevice, &type) );
        const char *nnVersion;
        CHECK_NNAPI_ERROR( ANeuralNetworksDevice_getVersion(nnDevice, &nnVersion) );
        SLOG_INFO << nnDeviceName << ',' << featureLevel << ',' << type << ',' << nnVersion << std::endl;
    }

    SLOG_INFO << "ANEURALNETWORKS_DEVICE_ACCELERATOR: " << ANEURALNETWORKS_DEVICE_ACCELERATOR << std::endl;
    SLOG_INFO << "ANEURALNETWORKS_DEVICE_CPU: " << ANEURALNETWORKS_DEVICE_CPU << std::endl;
    SLOG_INFO << "ANEURALNETWORKS_DEVICE_GPU: " << ANEURALNETWORKS_DEVICE_GPU << std::endl;
    SLOG_INFO << "ANEURALNETWORKS_DEVICE_OTHER: " << ANEURALNETWORKS_DEVICE_OTHER << std::endl;
    SLOG_INFO << "ANEURALNETWORKS_DEVICE_UNKNOWN: " << ANEURALNETWORKS_DEVICE_UNKNOWN << std::endl;
}


int main()
{
    getAndroidSdkVersion();
    getDevices();

    ANeuralNetworksModel *model = nullptr;
    ANeuralNetworksCompilation *compilation = nullptr;
    ANeuralNetworksExecution *execution = nullptr;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_create(&model) );


    std::map<std::string, uint32_t> operandIdxes;
    std::map<std::string, std::vector<uint32_t>> shapeIdxes;
    uint32_t opIdx = 0; // TODO: maybe don't need this

    // NNAPI default data layout NHWC
    const uint32_t NET_WIDTH = 300;
    const uint32_t NET_HEIGHT = 300;
    const uint32_t NET_CHANNELS = 3;
    const std::string INPUT_LAYER_NAME = "data";
    addTensor(model, operandIdxes, shapeIdxes, INPUT_LAYER_NAME, opIdx++, {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS});


    float dummyWeightBuf[16 * 5 * 5 * 3];
    float dummyBiasBuf[16];
    std::fill(dummyWeightBuf, dummyWeightBuf + (16 * 5 * 5 * 3), 2.0f);
    std::fill(dummyBiasBuf, dummyBiasBuf + 16, 1.0f);

    addTensor(model, operandIdxes, shapeIdxes, "conv2d_1_weight", opIdx++, {16, 5, 5, 3}, dummyWeightBuf);
    addTensor(model, operandIdxes, shapeIdxes, "conv2d_1_bias", opIdx++, {16}, dummyBiasBuf);

    conv2d(model, operandIdxes, shapeIdxes, "conv2d_1",
        opIdx, "data", "conv2d_1_weight", "conv2d_1_bias",
        2, 2, 2, 2, 1, 1, ANEURALNETWORKS_FUSED_NONE, false, 1, 1,
        "conv2d_1_out");











    // Identify the input and output tensors to the model. TODO: Ugly here
    std::vector<uint32_t> INPUT_INDICES = {operandIdxes.at("data")};
    std::vector<uint32_t> OUTPUT_INDICES = {operandIdxes.at("conv2d_1_out")};

    const uint32_t NET_IN_SIZE1 = std::accumulate(shapeIdxes.at("data").begin(), shapeIdxes.at("data").end(), 1, std::multiplies<uint32_t>());
    const uint32_t NET_OUT_SIZE1 = std::accumulate(shapeIdxes.at("conv2d_1_out").begin(), shapeIdxes.at("conv2d_1_out").end(), 1, std::multiplies<uint32_t>());
    SLOG_INFO << "NET_IN_SIZE1: " << NET_IN_SIZE1 << std::endl;
    SLOG_INFO << "NET_OUT_SIZE1: " << NET_OUT_SIZE1 << std::endl;

    float *indata1 = new float[NET_IN_SIZE1];
    std::fill(indata1, indata1 + NET_IN_SIZE1, 1.0f);
    std::vector<float *> INPUT_DATA = {indata1};
    assert(INPUT_INDICES.size() == INPUT_DATA.size());
    std::vector<uint32_t> INPUT_SIZES = {NET_IN_SIZE1};
    assert(INPUT_INDICES.size() == INPUT_SIZES.size());

    int fdout1 = ASharedMemory_create("dont_know_what_this_is_for", NET_OUT_SIZE1 * sizeof(float));
    ANeuralNetworksMemory *nnOutMem1;
    CHECK_NNAPI_ERROR( ANeuralNetworksMemory_createFromFd(NET_OUT_SIZE1 * sizeof(float), PROT_READ | PROT_WRITE, fdout1, 0, &nnOutMem1) );
    std::vector<ANeuralNetworksMemory *> OUTPUT_DATA = {nnOutMem1};
    assert(OUTPUT_INDICES.size() == OUTPUT_DATA.size());
    std::vector<uint32_t> OUTPUT_SIZES = {NET_OUT_SIZE1};
    assert(OUTPUT_INDICES.size() == OUTPUT_SIZES.size());
    





    // Finish constructing the model
    // The values of constant and intermediate operands cannot be altered after the finish function is called
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_identifyInputsAndOutputs(model, INPUT_INDICES.size(), INPUT_INDICES.data(), OUTPUT_INDICES.size(), OUTPUT_INDICES.data()) );
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_finish(model) );
    // Create the ANeuralNetworksCompilation object for the constructed model
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_create(model, &compilation) );
    // Set the preference for the compilation, so that the runtime and drivers can make better decisions
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER) );
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_finish(compilation) );
    // Multiple concurrent execution instances could be created from the same compiled model.
    // This sample only uses one execution of the compiled model.
    CHECK_NNAPI_ERROR( ANeuralNetworksExecution_create(compilation, &execution) );
    // Tell the execution to associate input data to the model inputs.
    // Note that the index here uses the operand of the model input list, not all operand list
    for (size_t i = 0; i < INPUT_INDICES.size(); ++i)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksExecution_setInput(execution, static_cast<int32_t>(i), nullptr, INPUT_DATA[i], INPUT_SIZES[i] * sizeof(float)) ); // support only FP32 now
    }
    // Set the output tensor that will be filled by executing the model.
    // We use shared memory here to minimize the copies needed for getting the output data.
    // Note that the index here uses the operand of the model output list, not all operand list
    for (size_t i = 0; i < INPUT_INDICES.size(); ++i)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksExecution_setOutputFromMemory(execution, static_cast<int32_t>(i), nullptr, OUTPUT_DATA[i], 0, OUTPUT_SIZES[i] * sizeof(float)) );
    }








    // Start the execution of the model.
    // Note that the execution here is asynchronous, and an ANeuralNetworksEvent object will be created to monitor the status of the execution.
    ANeuralNetworksEvent *event = nullptr;
    CHECK_NNAPI_ERROR( ANeuralNetworksExecution_startCompute(execution, &event) );
    // Wait until the completion of the execution. This could be done on a different thread.
    // By waiting immediately, we effectively make this a synchronous call.
    CHECK_NNAPI_ERROR( ANeuralNetworksEvent_wait(event) );







    // Validate the results.
    float *outputTensorPtr = reinterpret_cast<float *>(mmap(nullptr, NET_OUT_SIZE1 * sizeof(float), PROT_READ, MAP_SHARED, fdout1, 0));

    for (int32_t idx = 0; idx < 10; ++idx)
    {
        SLOG_INFO << outputTensorPtr[idx] << std::endl;
    }

    munmap(outputTensorPtr, NET_OUT_SIZE1 * sizeof(float));







    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
    delete [] indata1;
    ANeuralNetworksMemory_free(nnOutMem1);
    close(fdout1);



    SLOG_INFO << "fin" << std::endl;
    return 0;
}