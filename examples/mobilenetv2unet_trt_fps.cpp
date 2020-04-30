#include "buffers.h"
#include "common.h"
#include "logger.h"
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



void plotResult(cv::Mat &im, const float *scoremap, const int NETH, const float SCORE_TH)
{
    const int PLANESIZE = im.cols * NETH;
    uint8_t *shiftsrc = im.data + (im.rows - NETH) * im.cols * 3;
    for (int i = 0; i < PLANESIZE; ++i)
    {
        if (scoremap[i] > SCORE_TH)
        {
            shiftsrc[i * 3 + 1] = 255;
        }
    }
}

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
    const float SCORE_TH = config["model"]["score_threshold"].get<float>();
    std::vector<std::string> OUT_TENSOR_NAMES = config["caffe"]["output_layer"].get<std::vector<std::string> >();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string IMPATH = config["evaluate"]["images_for_fps"].get<std::string>();
    const std::string CAFFETXT = config["caffe"]["prototxt"].get<std::string>();
    const std::string CAFFEBIN = config["caffe"]["caffemodel"].get<std::string>();
    const bool FP16MODE = config["caffe"]["fp16_mode"].get<bool>();



    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    nvinfer1::IBuilder *trtbuilder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    nvinfer1::INetworkDefinition *trtnetwork = trtbuilder->createNetwork();
    nvinfer1::IBuilderConfig *trtconfig = trtbuilder->createBuilderConfig();
    nvcaffeparser1::ICaffeParser *trtparser = nvcaffeparser1::createCaffeParser();

    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = trtparser->parse(CAFFETXT.c_str(), CAFFEBIN.c_str(), *trtnetwork, nvinfer1::DataType::kFLOAT);
    for (auto& s : OUT_TENSOR_NAMES)
    {
        trtnetwork->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    trtbuilder->setMaxBatchSize(1);
    trtconfig->setMaxWorkspaceSize(36_MiB);

    if (FP16MODE)
    {
        trtconfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // std::shared_ptr<nvinfer1::ICudaEngine> engine = 
    //     std::shared_ptr<nvinfer1::ICudaEngine>(trtbuilder->buildEngineWithConfig(*trtnetwork, *trtconfig), samplesCommon::InferDeleter());
    nvinfer1::ICudaEngine *engine = trtbuilder->buildEngineWithConfig(*trtnetwork, *trtconfig);


    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(engine, 1);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    float* intensorPtrR = static_cast<float*>(buffers.getHostBuffer("data"));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;

    const float PRE_DIV = 1.0f / 255;
    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        cv::Mat imnet = cv::imread(cv::format(IMPATH.c_str(), (t % 10) + 1));
        const uint8_t *shiftsrc = imnet.data + (imnet.rows - NETH) * NETW * 3; // pass upper area, which is mostly sky

        for (int i = 0; i < NET_PLANESIZE; ++i)
        {
            intensorPtrR[i] = shiftsrc[3 * i + 2] * PRE_DIV;
            intensorPtrG[i] = shiftsrc[3 * i + 1] * PRE_DIV;
            intensorPtrB[i] = shiftsrc[3 * i + 0] * PRE_DIV;
        }

        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        buffers.copyInputToDevice(); // Memcpy from host input buffers to device input buffers
        context->execute(1, buffers.getDeviceBindings().data());
        buffers.copyOutputToHost(); // Memcpy from device output buffers to host output buffers


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());

        const float* scoresTensor = static_cast<const float*>(buffers.getHostBuffer(OUT_TENSOR_NAMES[0]));
        plotResult(imnet, scoresTensor, NETH, SCORE_TH);
        cv::imwrite("result_" + std::to_string(t + 1) + ".jpg", imnet);
    }

    // gLogInfo << "MobilenetV2Unet FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;

    trtbuilder->destroy();
    trtnetwork->destroy();
    trtconfig->destroy();
    trtparser->destroy();
    nvcaffeparser1::shutdownProtobufLibrary();

    return 0;
}