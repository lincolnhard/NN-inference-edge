#include "NvCaffeParser.h"
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

#include "trt_buffers.h"
#include "trt_common.h"


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

    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const float MEANR = config["model"]["mean"]["R"].get<float>();
    const float MEANG = config["model"]["mean"]["G"].get<float>();
    const float MEANB = config["model"]["mean"]["B"].get<float>();
    const float STDR = config["model"]["std"]["R"].get<float>();
    const float STDG = config["model"]["std"]["G"].get<float>();
    const float STDB = config["model"]["std"]["B"].get<float>();
    const std::string CAFFETXT = config["caffe"]["prototxt"].get<std::string>();
    const std::string CAFFEBIN = config["caffe"]["caffemodel"].get<std::string>();
    std::vector<std::string> OUT_TENSOR_NAMES = config["caffe"]["output_layer"].get<std::vector<std::string> >();
    const bool FP16MODE = config["caffe"]["fp16_mode"].get<bool>();
    std::vector<int> NUM_VERTEX_BYCLASS = config["model"]["class_num_vertex"].get<std::vector<int> >();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();




    TrtLogger trtlogger;
    cudaSetDevice(0); // TX2 has only one device
    initLibNvInferPlugins(&trtlogger, "");

    std::shared_ptr<nvinfer1::IBuilder> trtbuilder =
        std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger), InferDeleter());
    std::shared_ptr<nvinfer1::INetworkDefinition> trtnetwork =
        std::shared_ptr<nvinfer1::INetworkDefinition>(trtbuilder->createNetwork(), InferDeleter());
    std::shared_ptr<nvinfer1::IBuilderConfig> trtconfig =
        std::shared_ptr<nvinfer1::IBuilderConfig>(trtbuilder->createBuilderConfig(), InferDeleter());
    std::shared_ptr<nvcaffeparser1::ICaffeParser> trtparser =
        std::shared_ptr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser(), InferDeleter());


    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = trtparser->parse(CAFFETXT.c_str(), CAFFEBIN.c_str(), *trtnetwork, nvinfer1::DataType::kFLOAT);

    for (auto& s : OUT_TENSOR_NAMES)
    {
        trtnetwork->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    trtbuilder->setMaxBatchSize(1);
    trtconfig->setMaxWorkspaceSize(300_MiB);

    if (FP16MODE)
    {
        trtconfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    std::shared_ptr<nvinfer1::ICudaEngine> engine = 
        std::shared_ptr<nvinfer1::ICudaEngine>(trtbuilder->buildEngineWithConfig(*trtnetwork, *trtconfig), InferDeleter());

    // transforming the engine into a format to store and use at a later time for inference
    nvinfer1::IHostMemory *serializedEngine = engine->serialize();
    std::ofstream engineFile("data/espnetv2_segfcos.engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    serializedEngine->destroy();
    engineFile.close();


    BufferManager buffers(engine, 1);

    std::shared_ptr<nvinfer1::IExecutionContext> context =
        std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext(), InferDeleter());

    float* intensorPtrR = static_cast<float*>(buffers.getHostBuffer("data"));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;




    cv::Mat im = cv::imread(IMPATH);
    cv::Mat imnet;
    cv::resize(im, imnet, cv::Size(NETW, NETH));
    for (int i = 0; i < NET_PLANESIZE; ++i)
    {
        intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
        intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
        intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
    }

    buffers.copyInputToDevice(); // Memcpy from host input buffers to device input buffers
    context->execute(1, buffers.getDeviceBindings().data());
    buffers.copyOutputToHost(); // Memcpy from device output buffers to host output buffers




    nvcaffeparser1::shutdownProtobufLibrary();

    return 0;
}