#include "builder.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"

#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>

#include <unistd.h>

static auto LOG = spdlog::stdout_color_mt("MODEL_BUILDER");

namespace gallopwave
{

void ModelBuilder::getSdkVersion()
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


void ModelBuilder::getDevices()
{
    uint32_t deviceCount;
    CHECK_NNAPI_ERROR( ANeuralNetworks_getDeviceCount(&deviceCount) );
    SLOG_INFO << "NNAPI device list: " << std::endl;
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

        SLOG_INFO << i << ": " << nnDeviceName << ',' << featureLevel << ',' << type << ',' << nnVersion << std::endl;
        devices.push_back(nnDevice);
    }
}


size_t ModelBuilder::getElementSize(OperandCode operandcode)
{
    switch (operandcode)
    {
        case ANEURALNETWORKS_TENSOR_FLOAT32:
            return 4;
        case ANEURALNETWORKS_TENSOR_INT32:
            return 4;
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            return 1;
        default:
            SLOG_ERROR << "Not supported operand type: " <<  operandcode << std::endl;
            exit(1);
    }
}


ModelBuilder::~ModelBuilder()
{
    for (auto it: outputTensors)
    {
        munmap(it.data, it.sizeBytes);
        ANeuralNetworksMemory_free(it.nnMemPtr);
        close(it.fd);
    }
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
}

ModelBuilder::ModelBuilder()
{
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_create(&model) );
    opIdx = 0;
}

void ModelBuilder::addTensor (std::string name,
                            std::vector<uint32_t> dims,
                            OperandCode operandcode,
                            const void *srcbuffer,
                            float scale,
                            int32_t zeroPoint
                            )
{
    ANeuralNetworksOperandType operandType;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(dims.size());
    operandType.dimensions = dims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name] = opIdx;
    shapeIdxes[name] = dims;

    if (srcbuffer != nullptr)
    {
        const size_t bytes = getElementSize(operandcode) * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, srcbuffer, bytes) );
    }
    ++opIdx;
}

void ModelBuilder::conv2d (const std::string& name,
                        const std::string& input,
                        const std::string& weight,
                        const std::string& bias,
                        OperandCode operandcode,
                        int32_t padLeft,
                        int32_t padRight,
                        int32_t padTop,
                        int32_t padBottom,
                        int32_t strideX,
                        int32_t strideY,
                        bool isDepthWise,
                        FuseCode fusecode,
                        const std::string& output,
                        float scale,
                        int32_t zeroPoint
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
    operandIdxes[name + "_padLeft"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padLeft, sizeof(padLeft)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padRight"] = opIdx;
    CHECK_NNAPI_ERROR ( ANeuralNetworksModel_setOperandValue(model, opIdx, &padRight, sizeof(padRight)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padTop"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padTop, sizeof(padTop)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padBottom"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padBottom, sizeof(padBottom)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideX"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &strideX, sizeof(strideX)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideY"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &strideY, sizeof(strideY)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    if (isDepthWise)
    {
        int32_t depthMultiplier = 1; // TODO: Support only for depth_multiplier == 1 now
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
        operandIdxes[name + "_depthMultiplier"] = opIdx;
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &depthMultiplier, sizeof(depthMultiplier)) );
        parameterIdxes.push_back(opIdx);
        ++opIdx;
    }


    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_activation"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &fusecode, sizeof(fusecode)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    const auto inDims = shapeIdxes.at(input);
    const auto wDims = shapeIdxes.at(weight);
    const uint32_t outN = inDims[0];
    const uint32_t outH = (inDims[1] - wDims[1] + padTop + padBottom) / strideY + 1;
    const uint32_t outW = (inDims[2] - wDims[2] + padLeft + padRight) / strideX + 1;
    uint32_t outC = wDims[0];

    if (isDepthWise)
    {
        assert(inDims[3] == wDims[3]); // TODO: Support only for depth_multiplier == 1 now
        outC = wDims[3];
    }

    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(inDims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;

    outIdxes.push_back(opIdx);
    ++opIdx;

    if (isDepthWise)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_DEPTHWISE_CONV_2D, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
    }
    else
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
    }

}


void ModelBuilder::eltwise (const std::string& name,
                            const std::string& input1,
                            const std::string& input2,
                            FuseCode fusecode,
                            const std::string& output,
                            OperandCode operandcode,
                            EltwiseCode eltwisecode,
                            float scale,
                            int32_t zeroPoint
                            )
{
    std::vector<uint32_t> parameterIdxes;

    const auto input1Idx = operandIdxes.at(input1);
    const auto input2Idx = operandIdxes.at(input2);
    parameterIdxes.push_back(input1Idx);
    parameterIdxes.push_back(input2Idx);

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = nullptr;
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_activation"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &fusecode, sizeof(fusecode)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    const auto in1Dims = shapeIdxes.at(input1);
    const auto in2Dims = shapeIdxes.at(input2);
    const uint32_t outN = (in1Dims[0] > in2Dims[0] ? in1Dims[0] : in2Dims[0]);
    const uint32_t outH = (in1Dims[1] > in2Dims[1] ? in1Dims[1] : in2Dims[1]);
    const uint32_t outW = (in1Dims[2] > in2Dims[2] ? in1Dims[2] : in2Dims[2]);
    const uint32_t outC = (in1Dims[3] > in2Dims[3] ? in1Dims[3] : in2Dims[3]);
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(in1Dims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;

    outIdxes.push_back(opIdx);
    ++opIdx;

    if (eltwisecode == ELTWISE_ADDITION)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
    }
    else if (eltwisecode == ELTWISE_MULTIPLY)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
    }
}


void ModelBuilder::maxpool(const std::string& name,
                        const std::string& input,
                        OperandCode operandcode,
                        int32_t padLeft,
                        int32_t padRight,
                        int32_t padTop,
                        int32_t padBottom,
                        int32_t strideX,
                        int32_t strideY,
                        int32_t kernelW,
                        int32_t kernelH,
                        FuseCode fusecode,
                        const std::string& output,
                        float scale,
                        int32_t zeroPoint
                        )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    parameterIdxes.push_back(inputIdx);

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = nullptr;
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType));
    operandIdxes[name + "_padLeft"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padLeft, sizeof(padLeft)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padRight"] = opIdx;
    CHECK_NNAPI_ERROR ( ANeuralNetworksModel_setOperandValue(model, opIdx, &padRight, sizeof(padRight)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padTop"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padTop, sizeof(padTop)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_padBottom"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &padBottom, sizeof(padBottom)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideX"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &strideX, sizeof(strideX)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_strideY"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &strideY, sizeof(strideY)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_kernelW"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &kernelW, sizeof(kernelW)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_kernelH"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &kernelH, sizeof(kernelH)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_activation"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &fusecode, sizeof(fusecode)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    const auto inDims = shapeIdxes.at(input);
    const uint32_t outN = inDims[0];
    const uint32_t outH = (inDims[1] - kernelH + padTop + padBottom) / strideY + 1;
    const uint32_t outW = (inDims[2] - kernelW + padLeft + padRight) / strideX + 1;
    const uint32_t outC = inDims[3];
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(inDims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;

    outIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
}


void ModelBuilder::reduce(const std::string& name,
                        const std::string& input,
                        OperandCode operandcode,
                        const std::string& output,
                        float scale,
                        int32_t zeroPoint
                        )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    parameterIdxes.push_back(inputIdx);

    std::vector<uint32_t> axis = {1, 2}; // Reduce H, W channels

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_INT32;
    operandType.dimensionCount = static_cast<uint32_t>(axis.size());
    operandType.dimensions = axis.data();
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_axis"] = opIdx;
    parameterIdxes.push_back(opIdx);
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, reinterpret_cast<int32_t *>(axis.data()), sizeof(int32_t) * axis.size()) );
    ++opIdx;

    int32_t keepDims = 1;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = nullptr;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType));
    operandIdxes[name + "_keep"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &keepDims, sizeof(keepDims)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;


    const auto inDims = shapeIdxes.at(input);
    const uint32_t outN = inDims[0];
    const uint32_t outH = 1;
    const uint32_t outW = 1;
    const uint32_t outC = inDims[3];
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(outDims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;

    outIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MEAN, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
}


void ModelBuilder::sigmoid(const std::string& name,
                        const std::string& input,
                        OperandCode operandcode,
                        const std::string& output
                        )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    parameterIdxes.push_back(inputIdx);

    const auto inDims = shapeIdxes.at(input);
    const uint32_t outN = inDims[0];
    const uint32_t outH = inDims[1];
    const uint32_t outW = inDims[2];
    const uint32_t outC = inDims[3];
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    ANeuralNetworksOperandType operandType;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(outDims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = 1.0f / 256;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;
    outIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_LOGISTIC, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
}


void ModelBuilder::resize(const std::string& name,
                        const std::string& input,
                        int32_t outputW,
                        int32_t outputH,
                        OperandCode operandcode,
                        const std::string& output,
                        float scale,
                        int32_t zeroPoint
                        )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    parameterIdxes.push_back(inputIdx);

    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = nullptr;
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType));
    operandIdxes[name + "_outW"] = opIdx;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_setOperandValue(model, opIdx, &outputW, sizeof(outputW)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );
    operandIdxes[name + "_outH"] = opIdx;
    CHECK_NNAPI_ERROR ( ANeuralNetworksModel_setOperandValue(model, opIdx, &outputH, sizeof(outputH)) );
    parameterIdxes.push_back(opIdx);
    ++opIdx;


    const auto inDims = shapeIdxes.at(input);
    const uint32_t outN = inDims[0];
    const uint32_t outH = outputW;
    const uint32_t outW = outputH;
    const uint32_t outC = inDims[3];
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(outDims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;

    outIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_RESIZE_BILINEAR, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
}


void ModelBuilder::dequantize(const std::string& name,
                        const std::string& input,
                        OperandCode operandcode,
                        const std::string& output,
                        float scale,
                        int32_t zeroPoint
                        )
{
    std::vector<uint32_t> parameterIdxes;

    const auto inputIdx = operandIdxes.at(input);
    parameterIdxes.push_back(inputIdx);

    const auto inDims = shapeIdxes.at(input);
    const uint32_t outN = inDims[0];
    const uint32_t outH = inDims[1];
    const uint32_t outW = inDims[2];
    const uint32_t outC = inDims[3];
    std::vector<uint32_t> outDims = {outN, outH, outW, outC};

    std::vector<uint32_t> outIdxes;
    ANeuralNetworksOperandType operandType;
    operandType.type = operandcode;
    operandType.dimensionCount = static_cast<uint32_t>(outDims.size());
    operandType.dimensions = outDims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zeroPoint;
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperand(model, &operandType) );

    operandIdxes[output] = opIdx;
    shapeIdxes[output] = outDims;
    outIdxes.push_back(opIdx);
    ++opIdx;

    CHECK_NNAPI_ERROR( ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_DEQUANTIZE, parameterIdxes.size(), &parameterIdxes[0], outIdxes.size(), &outIdxes[0]) );
}


void ModelBuilder::setInputTensors(std::string name,
                                void* dataptr,
                                OperandCode operandcode
                                )
{
    uint32_t idx = operandIdxes.at(name);
    std::vector<uint32_t> shape = shapeIdxes.at(name);
    uint32_t sizebyte = getElementSize(operandcode) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    inputTensors.push_back({idx, shape, sizebyte, name, dataptr});
}


void ModelBuilder::setOutputTensors(std::string name,
                                    OperandCode operandcode
                                    )
{
    uint32_t idx = operandIdxes.at(name);
    std::vector<uint32_t> shape = shapeIdxes.at(name);
    uint32_t sizebyte = getElementSize(operandcode) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    int fd = ASharedMemory_create("an_optional_name", sizebyte);
    ANeuralNetworksMemory *memptr = nullptr;
    CHECK_NNAPI_ERROR( ANeuralNetworksMemory_createFromFd(sizebyte, PROT_READ | PROT_WRITE, fd, 0, &memptr) );
    void* dataptr = mmap(nullptr, sizebyte, PROT_READ, MAP_SHARED, fd, 0);
    outputTensors.push_back({idx, shape, sizebyte, name, dataptr, fd, memptr});
}

void ModelBuilder::compile (int32_t dIdx)
{
    std::vector<uint32_t> inputIndices;
    std::vector<uint32_t> outputIndices;
    for (auto it: inputTensors)
    {
        inputIndices.push_back(it.index);
    }
    for (auto it: outputTensors)
    {
        outputIndices.push_back(it.index);
    }
    // The values of constant and intermediate operands cannot be altered after the finish function is called
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_identifyInputsAndOutputs(model, inputIndices.size(), inputIndices.data(), outputIndices.size(), outputIndices.data()) );
    CHECK_NNAPI_ERROR( ANeuralNetworksModel_finish(model) );
    if (dIdx != -1)
    {
        // TODO: Here use only one device :)
        ANeuralNetworksDevice *devicePtr = devices[dIdx];
        bool supportedOps[140];
        for (int i = 0; i < 123; ++i)
        {
            supportedOps[i] = false;
        }
        CHECK_NNAPI_ERROR( ANeuralNetworksModel_getSupportedOperationsForDevices(model, &devicePtr, 1, supportedOps) );
        for (int i = 0; i < 123; ++i)
        {
            SLOG_WARN << supportedOps[i] << std::endl;
        }
        CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_createForDevices(model, &devicePtr, 1, &compilation) );
    }
    else
    {
        // auto select
        CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_create(model, &compilation) );
    }
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER) );
    CHECK_NNAPI_ERROR( ANeuralNetworksCompilation_finish(compilation) );
}

void ModelBuilder::execute (void)
{
    // Multiple concurrent execution instances could be created from the same compiled model.
    CHECK_NNAPI_ERROR( ANeuralNetworksExecution_create(compilation, &execution) );
    // Associate to the model inputs. Note that the index here uses the operand of the model input list, not all operand list
    for (size_t i = 0; i < inputTensors.size(); ++i)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksExecution_setInput(execution, static_cast<int32_t>(i), nullptr, inputTensors[i].data, inputTensors[i].sizeBytes) );
    }
    // Set the output tensor that will be filled by executing the model. Shared memory here to minimize the copies needed for getting the output data.
    // Note that the index here uses the operand of the model output list, not all operand list
    for (size_t i = 0; i < outputTensors.size(); ++i)
    {
        CHECK_NNAPI_ERROR( ANeuralNetworksExecution_setOutputFromMemory(execution, static_cast<int32_t>(i), nullptr, outputTensors[i].nnMemPtr, 0, outputTensors[i].sizeBytes) );
    }
    // Note that the execution here is asynchronous, event will be created to monitor the status of the execution.
    CHECK_NNAPI_ERROR( ANeuralNetworksExecution_startCompute(execution, &event) );
    // Wait until the completion of the execution. This could be done on a different thread.
    // By waiting immediately, we effectively make this a synchronous call.
    CHECK_NNAPI_ERROR( ANeuralNetworksEvent_wait(event) );

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
}

std::vector<void *> ModelBuilder::getOutput(void)
{
    std::vector<void *> outputTensorPtrs;
    for (auto it: outputTensors)
    {
        outputTensorPtrs.push_back(it.data);
    }

    return outputTensorPtrs;
}



}

