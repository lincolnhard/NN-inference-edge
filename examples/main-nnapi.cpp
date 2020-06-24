#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <functional>
#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <unistd.h>

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
                std::string name,
                uint32_t index,
                std::vector<uint32_t> dims,
                float scale = 0.0f,
                int32_t zeroPoint = 0,
                const void *srcbuffer = nullptr
                )
{
    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_FLOAT32; // support only FP32 now
    operandType.dimensionCount = static_cast<uint32_t>(dims.size());
    operandType.dimensions = dims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name] = index;

    if (srcbuffer != nullptr)
    {
        size_t elementSize = 4; // support only FP32 now
        const size_t bytes = elementSize * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
        ANeuralNetworksModel_setOperandValue(model, index, srcbuffer, bytes);
    }
}

void conv2d (ANeuralNetworksModel *model,
            std::map<std::string, uint32_t>& operandIdxes,
            std::string name,
            uint32_t& index,
            const std::string &input,
            const std::string &weight,
            const std::string &bias,
            const std::string &output,
            const int32_t padLeft,
            const int32_t padRight,
            const int32_t padTop,
            const int32_t padBottom,
            const int32_t strideX,
            const int32_t strideY,
            const int32_t dilationX,
            const int32_t dilationY
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
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensions = {};
    operandType.scale = 0.0;
    operandType.zeroPoint = 0;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name + "_padLeft"] = index;
    ANeuralNetworksModel_setOperandValue(model, index, &padLeft, sizeof(padLeft));
    ++index;

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensions = {};
    operandType.scale = 0.0;
    operandType.zeroPoint = 0;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name + "_padRight"] = index;
    ANeuralNetworksModel_setOperandValue(model, index, &padRight, sizeof(padRight));
    ++index;

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensions = {};
    operandType.scale = 0.0;
    operandType.zeroPoint = 0;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name + "_padTop"] = index;
    ANeuralNetworksModel_setOperandValue(model, index, &padTop, sizeof(padTop));
    ++index;

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensions = {};
    operandType.scale = 0.0;
    operandType.zeroPoint = 0;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name + "_padBottom"] = index;
    ANeuralNetworksModel_setOperandValue(model, index, &padBottom, sizeof(padBottom));
    ++index;

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensions = {};
    operandType.scale = 0.0;
    operandType.zeroPoint = 0;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name + "_strideX"] = index;
    ANeuralNetworksModel_setOperandValue(model, index, &strideX, sizeof(strideX));
    ++index;

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensions = {};
    operandType.scale = 0.0;
    operandType.zeroPoint = 0;
    ANeuralNetworksModel_addOperand(model, &operandType);
    operandIdxes[name + "_strideY"] = index;
    ANeuralNetworksModel_setOperandValue(model, index, &strideY, sizeof(strideY));
    ++index;

}

int main()
{
    int32_t status = 0;
    ANeuralNetworksModel *model;
    status = ANeuralNetworksModel_create(&model);
    CHECK_NNAPI_ERROR(status);


    std::map<std::string, uint32_t> operandIdxes;

    uint32_t opIdx = 0;
    addTensor(model, operandIdxes, "data", opIdx++, {1, 480, 640, 3});

    float dummyWeightBuf[1000];
    float dummyBiasBuf[1000];

    addTensor(model, operandIdxes, "weight", opIdx++, {1, 480, 640, 3}, 0.0f, 0, dummyWeightBuf);
    addTensor(model, operandIdxes, "bias", opIdx++, {1, 480, 640, 3}, 0.0f, 0, dummyBiasBuf);


    ANeuralNetworksModel_free(model);
    return 0;
}