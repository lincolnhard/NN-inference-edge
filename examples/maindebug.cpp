#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "resize_nearest.h"
#include "resize_bilinear.h"
#include "argmax.h"
#include "reshape.h"
#include "trt_common.h"

class TrtLogger : public nvinfer1::ILogger 
{
   public:
    TrtLogger();
    /// Log message
    /// \param[in] severity, Severity of the message
    /// \param[in] msg, Message to be logged
    void log(Severity severity, char const *msg) override;

    /// Set if logging is verbose
    /// \param[in] verbose If true, logging will be more verbose
    void SetVerbose(bool const verbose);

   private:
    bool verbose_;
};

TrtLogger::TrtLogger() : verbose_(false) {}

void TrtLogger::log(nvinfer1::ILogger::Severity severity, char const *msg)
{
    if (verbose_)
    {
        std::cout << msg << std::endl;
    }
    switch (severity)
    {
        case Severity::kVERBOSE:
        case Severity::kINFO:
            LOG(INFO) << msg << std::endl;
            break;
        case Severity::kWARNING:
            LOG(WARNING) << msg << std::endl;
            break;
        case Severity::kERROR:
            LOG(ERROR) << msg << std::endl;
            break;
        case Severity::kINTERNAL_ERROR:
            LOG(FATAL) << msg << std::endl;
            break;
    }
}

void TrtLogger::SetVerbose(bool const verbose) { verbose_ = verbose; }

int main(int argc, char *argv[])
{
    // google::InitGoogleLogging(argv[0]);

    TrtLogger trtlogger;
    cudaSetDevice(0); // TX2 has only one device
    initLibNvInferPlugins(&trtlogger, "");

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(trtlogger);
    nvinfer1::INetworkDefinition *network = builder->createNetwork();
    nvuffparser::IUffParser *parser = nvuffparser::createUffParser();


    std::vector<std::pair<std::string, nvinfer1::Dims>> inputLayers;
    std::string inputLayerName = "transpose";
    nvinfer1::Dims inputLayerDims;
    inputLayerDims.nbDims = 3;
    inputLayerDims.d[0] = 3;
    inputLayerDims.d[1] = 512;
    inputLayerDims.d[2] = 512;
    inputLayers.push_back(std::make_pair(inputLayerName, inputLayerDims));


    std::vector<std::string> outputLayers;
    outputLayers.push_back("NMS");
    outputLayers.push_back("ArgMax");

    std::string uffPath = "/home/gw/Documents/NN-inference-using-SNPE/data/uninet1.uff";

    for (auto &s : inputLayers)
    {
        nvuffparser::UffInputOrder order = nvuffparser::UffInputOrder::kNCHW;
        if (!parser->registerInput(s.first.c_str(), s.second, order))
        {
            LOG(WARNING) << "Failed to register input " << s.first << std::endl;
        }
    }

    for (auto &s : outputLayers)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            LOG(WARNING) << "Failed to register output " << s << std::endl;
        }
    }

    parser->parse(uffPath.c_str(), *network, nvinfer1::DataType::kFLOAT);


    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(168_MiB);



    engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
    {
        LOG(WARNING) << "could not build engine" << std::endl;
    }
    else
    {
        nvinfer1::IHostMemory *serializedEngine = engine->serialize();
        std::ofstream engineFile("data/uninet1.engine", std::ios::binary);
        engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
        serializedEngine->destroy();
        engineFile.close();
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    engine->destroy();

    return 0;
}
