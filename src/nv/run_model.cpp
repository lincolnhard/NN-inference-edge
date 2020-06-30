#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "log_stream.hpp"
#include "run_model.hpp"
#include "buffers.hpp"



static auto LOG = spdlog::stdout_color_mt("NV_RUN");



#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            SLOG_ERROR << "Cuda failure: " << ret << std::endl;                                                        \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)




void gallopwave::NVLogger::log(Severity severity, char const *msg)
{
    switch (severity)
    {
        case Severity::kVERBOSE:
        case Severity::kINFO:
            SLOG_INFO << msg << std::endl;
            break;
        case Severity::kWARNING:
            SLOG_WARN << msg << std::endl;
            break;
        case Severity::kERROR:
        case Severity::kINTERNAL_ERROR:
            SLOG_ERROR << msg << std::endl;
            break;
    }
}

gallopwave::NVModel::~NVModel(void)
{
    nvcaffeparser1::shutdownProtobufLibrary();
    nvuffparser::shutdownProtobufLibrary();
    cudaStreamDestroy(stream);
}

gallopwave::NVModel::NVModel(std::string onnxPath, bool isFP16)
{
    initLibNvInferPlugins(&logger, "");
    auto builder = NVUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = NVUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    auto config = NVUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = NVUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    auto parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));
    if (!parsed)
    {
        SLOG_ERROR << "Failed to parse assigned ONNX file" << std::endl;
        abort();
    }

    builder->setMaxBatchSize(1); // TODO: tune here
    config->setMaxWorkspaceSize(100_MiB);

    if (isFP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    engine = NVUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    context = NVUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    initBuffers();
}

gallopwave::NVModel::NVModel(std::string prototxtPath,
                            std::string caffemodelPath,
                            std::vector<std::string>& outTensorNames,
                            bool isFP16)
{
    initLibNvInferPlugins(&logger, "");
    auto builder = NVUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = NVUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    auto config = NVUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = NVUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());

    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = 
            parser->parse(prototxtPath.c_str(), caffemodelPath.c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s: outTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    builder->setMaxBatchSize(1); // TODO: tune here
    config->setMaxWorkspaceSize(36_MiB);

    if (isFP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    engine = NVUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    context = NVUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    initBuffers();
}


gallopwave::NVModel::NVModel(std::string uffPath,
                            std::vector<std::pair<std::string, nvinfer1::Dims>>& inTensorNamesShapes,
                            std::vector<std::string>& outTensorNames,
                            bool isFP16)
{
    cudaSetDevice(0); // TX2 has only one device
    initLibNvInferPlugins(&logger, "");
    auto builder = NVUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = NVUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    auto config = NVUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = NVUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

    // specify which tensors are inputs (and their dimensions)
    for (auto const &s : inTensorNamesShapes)
    {
        nvuffparser::UffInputOrder order = nvuffparser::UffInputOrder::kNCHW;
        if (!parser->registerInput(s.first.c_str(), s.second, order))
        {
            SLOG_ERROR << "Failed to register input " << s.first << std::endl;
            abort();
        }
    }

    // specify which tensors are outputs
    for (auto const &s : outTensorNames)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            SLOG_ERROR << "Failed to register output " << s << std::endl;
            abort();
        }
    }

    if (!parser->parse(uffPath.c_str(), *network, isFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT))
    {
        SLOG_ERROR << "Failed in uff parsing step" << std::endl;
        abort();
    }

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(36_MiB);
    if (isFP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    engine = NVUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    context = NVUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    initBuffers();
}


gallopwave::NVModel::NVModel(std::string enginePath)
{
    initLibNvInferPlugins(&logger, "");
    std::ifstream engineFile(enginePath, std::ios::binary);
    std::vector<char> engineFileStream(std::istreambuf_iterator<char>(engineFile), {});

    auto runtime = NVUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    engine = NVUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineFileStream.data(),
                                                                                engineFileStream.size(),
                                                                                nullptr));

    context = NVUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    engineFile.close();

    initBuffers();
}


void gallopwave::NVModel::outputEngine(std::string dstPath)
{
    auto serializedEngine = NVUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
    std::ofstream engineFile(dstPath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
}

void gallopwave::NVModel::initBuffers(void)
{
    // Create host and device buffers
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        auto dims = engine->getBindingDimensions(i);
        int vecDim = engine->getBindingVectorizedDim(i);
        assert(vecDim == -1); // TODO: need check

        SLOG_INFO << engine->getBindingName(i) << ": " << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ", " << dims.d[3] << std::endl;

        size_t vol = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        nvinfer1::DataType type = engine->getBindingDataType(i);

        std::unique_ptr<ManagedBuffer> managedBuf{new ManagedBuffer()};
        managedBuf->deviceBuffer = DeviceBuffer(vol, type);
        managedBuf->hostBuffer = HostBuffer(vol, type);

        deviceBindings.emplace_back(managedBuf->deviceBuffer.data());
        managedBuffers.emplace_back(std::move(managedBuf));
    }

    CHECK(cudaStreamCreate(&stream));
}


void* gallopwave::NVModel::getBuffer(const bool isHost, const std::string& tensorName) const
{
    int index = engine->getBindingIndex(tensorName.c_str());
    if (index == -1)
    {
        SLOG_WARN << "No assigned i/o tensor name" << std::endl;
        return nullptr;
    }

    return (isHost ? managedBuffers[index]->hostBuffer.data() : managedBuffers[index]->deviceBuffer.data());
}


void* gallopwave::NVModel::getDeviceBuffer(const std::string& tensorName) const
{
    return getBuffer(false, tensorName);
}


void* gallopwave::NVModel::getHostBuffer(const std::string& tensorName) const
{
    return getBuffer(true, tensorName);
}


void gallopwave::NVModel::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async)
{
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        void* dstPtr = deviceToHost ? managedBuffers[i]->hostBuffer.data() : managedBuffers[i]->deviceBuffer.data();
        const void* srcPtr = deviceToHost ? managedBuffers[i]->deviceBuffer.data() : managedBuffers[i]->hostBuffer.data();
        const size_t byteSize = managedBuffers[i]->hostBuffer.nbBytes();
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        if ((copyInput && engine->bindingIsInput(i)) || (!copyInput && !engine->bindingIsInput(i)))
        {
            if (async)
            {
                CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}

void gallopwave::NVModel::copyInputToDevice()
{
    memcpyBuffers(true, false, false);
}


void gallopwave::NVModel::copyOutputToHost()
{
    memcpyBuffers(false, true, false);
}


void gallopwave::NVModel::copyInputToDeviceAsync()
{
    memcpyBuffers(true, false, true);
}


void gallopwave::NVModel::copyOutputToHostAsync()
{
    memcpyBuffers(false, true, true);
}


void gallopwave::NVModel::run(void)
{
    copyInputToDevice(); // Memcpy from host input buffers to device input buffers
    context->execute(1, deviceBindings.data()); // Synchronously execute inference on a batch (batch = 1 here)
    copyOutputToHost(); // Memcpy from device output buffers to host output buffers
}

void gallopwave::NVModel::runAsync(void)
{
    copyInputToDeviceAsync();
    context->enqueue(1, deviceBindings.data(), stream, nullptr);
    copyOutputToHostAsync();
    cudaStreamSynchronize(stream); // Wait for the work in the stream to complete
}