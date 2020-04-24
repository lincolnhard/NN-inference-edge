#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvUffParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>

#include <json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;



int main(int ac, char *av[])
{
    if (ac != 2)
    {
        gLogError << av[0] << " [config_file.json]" << std::endl;
        return 1;
    }

    nlohmann::json config;
    std::ifstream fin;
    fin.open(av[1]);
    fin >> config;
    fin.close();



    const std::string IMPATH = config["evaluate"]["images_for_fps"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const std::string TRT_UFF_PATH = config["trt"]["uff"].get<std::string>();




    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    auto trtbuilder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    auto trtnetwork = SampleUniquePtr<nvinfer1::INetworkDefinition>(trtbuilder->createNetwork());
    auto trtconfig = SampleUniquePtr<nvinfer1::IBuilderConfig>(trtbuilder->createBuilderConfig());
    auto trtparser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

    // parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
    // parser->registerOutput(mParams.outputTensorNames[0].c_str());
    trtparser->parse(TRT_UFF_PATH.c_str(), *trtnetwork, nvinfer1::DataType::kFLOAT);
    trtbuilder->setMaxBatchSize(1);
    trtconfig->setMaxWorkspaceSize(1_GiB);

    std::shared_ptr<nvinfer1::ICudaEngine> engine = 
        std::shared_ptr<nvinfer1::ICudaEngine>(trtbuilder->buildEngineWithConfig(*trtnetwork, *trtconfig), samplesCommon::InferDeleter());

    // std::cout << engine->getBindingDimensions(0).d[0] << std::endl;
    // std::cout << engine->getBindingDimensions(0).d[1] << std::endl;
    // std::cout << engine->getBindingDimensions(0).d[2] << std::endl;


    nvuffparser::shutdownProtobufLibrary();

    return 0;
}