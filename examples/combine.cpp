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

#include "postprocess_fcos.hpp"


template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

int main(int ac, char *av[])
{
    const int NETW = 640;
    const int NETH = 384;
    const int NET_PLANESIZE = NETW * NETH;
    const std::string IMPATH = "pics/%d_net_big.jpg";
    const int EVALUATE_TIMES = 10;
    const float MEANR = 0.3856;
    const float MEANG = 0.3856;
    const float MEANB = 0.3856;
    const float STDR = 0.3856;
    const float STDG = 0.3856;
    const float STDB = 0.3856;
    const float SCORE_TH = 0.0f;
    const std::string CAFFETXT_FCOS = "data/mnasneta1fcos.prototxt";
    const std::string CAFFEBIN_FCOS = "data/mnasneta1fcos.caffemodel";
    const std::string CAFFETXT_UNET = "data/lane0125.prototxt";
    const std::string CAFFEBIN_UNET = "data/lane0125.caffemodel";

    std::vector<std::string> OUT_TENSOR_NAMES_FCOS;
    OUT_TENSOR_NAMES_FCOS.push_back("scoremap_perm");
    OUT_TENSOR_NAMES_FCOS.push_back("centernessmap_perm");
    OUT_TENSOR_NAMES_FCOS.push_back("regressionmap_perm");
    OUT_TENSOR_NAMES_FCOS.push_back("occlusionmap_perm");
    std::vector<std::string> OUT_TENSOR_NAMES_UNET;
    OUT_TENSOR_NAMES_UNET.push_back("conv18");

    const bool FP16MODE_FCOS = false;
    const bool FP16MODE_UNET = false;

    // std::vector<int> NUM_VERTEX_BYCLASS = fcosconfig["model"]["class_num_vertex"].get<std::vector<int> >();
    
    

    // PostprocessFCOS postprocesser(fcosconfig["model"]);



    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    auto trtbuilder1 = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    auto trtnetwork1 = SampleUniquePtr<nvinfer1::INetworkDefinition>(trtbuilder1->createNetwork());
    auto trtconfig1 = SampleUniquePtr<nvinfer1::IBuilderConfig>(trtbuilder1->createBuilderConfig());
    auto trtparser1 = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor1 = trtparser1->parse(CAFFETXT_FCOS.c_str(), CAFFEBIN_FCOS.c_str(), *trtnetwork1, nvinfer1::DataType::kFLOAT);
    for (auto& s : OUT_TENSOR_NAMES_FCOS)
    {
        trtnetwork1->markOutput(*blobNameToTensor1->find(s.c_str()));
    }
    trtbuilder1->setMaxBatchSize(1);
    trtconfig1->setMaxWorkspaceSize(36_MiB);
    if (FP16MODE_FCOS)
    {
        trtconfig1->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine1 = 
        std::shared_ptr<nvinfer1::ICudaEngine>(trtbuilder1->buildEngineWithConfig(*trtnetwork1, *trtconfig1), samplesCommon::InferDeleter());
    samplesCommon::BufferManager buffers_FCOS(engine1, 1);
    auto context_FCOS = SampleUniquePtr<nvinfer1::IExecutionContext>(engine1->createExecutionContext());
    float* intensorPtrR1 = static_cast<float*>(buffers_FCOS.getHostBuffer("data"));
    float* intensorPtrG1 = intensorPtrR1 + NET_PLANESIZE;
    float* intensorPtrB1 = intensorPtrG1 + NET_PLANESIZE;



    auto trtbuilder2 = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    auto trtnetwork2 = SampleUniquePtr<nvinfer1::INetworkDefinition>(trtbuilder2->createNetwork());
    auto trtconfig2 = SampleUniquePtr<nvinfer1::IBuilderConfig>(trtbuilder2->createBuilderConfig());
    auto trtparser2 = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor2 = trtparser2->parse(CAFFETXT_UNET.c_str(), CAFFEBIN_UNET.c_str(), *trtnetwork2, nvinfer1::DataType::kFLOAT);
    for (auto& s : OUT_TENSOR_NAMES_UNET)
    {
        trtnetwork2->markOutput(*blobNameToTensor2->find(s.c_str()));
    }
    trtbuilder2->setMaxBatchSize(1);
    trtconfig2->setMaxWorkspaceSize(36_MiB);
    if (FP16MODE_UNET)
    {
        trtconfig2->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine2 = 
        std::shared_ptr<nvinfer1::ICudaEngine>(trtbuilder2->buildEngineWithConfig(*trtnetwork2, *trtconfig2), samplesCommon::InferDeleter());
    samplesCommon::BufferManager buffers_UNET(engine2, 1);
    auto context_UNET = SampleUniquePtr<nvinfer1::IExecutionContext>(engine2->createExecutionContext());
    float* intensorPtrR2 = static_cast<float*>(buffers_UNET.getHostBuffer("data"));
    float* intensorPtrG2 = intensorPtrR2 + NET_PLANESIZE;
    float* intensorPtrB2 = intensorPtrG2 + NET_PLANESIZE;



    std::thread fcosthread([&](){
        double timesum = 0.0;
        for (int t = 0; t < EVALUATE_TIMES; ++t)
        {
            cv::Mat imnet = cv::imread(cv::format(IMPATH.c_str(), (t % 10) + 1));
            for (int i = 0; i < NET_PLANESIZE; ++i)
            {
                intensorPtrR1[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
                intensorPtrG1[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
                intensorPtrB1[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
            }

            std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

            buffers_FCOS.copyInputToDevice();
            context_FCOS->execute(1, buffers_FCOS.getDeviceBindings().data());
            buffers_FCOS.copyOutputToHost();

            const float* scoresTensor = static_cast<const float*>(buffers_FCOS.getHostBuffer("scoremap_perm"));
            const float* centernessTensor = static_cast<const float*>(buffers_FCOS.getHostBuffer("centernessmap_perm"));
            const float* vertexTensor = static_cast<const float*>(buffers_FCOS.getHostBuffer("regressionmap_perm"));
            const float* occlusionsTensor = static_cast<const float*>(buffers_FCOS.getHostBuffer("occlusionmap_perm"));
            std::vector<const float *> featuremaps {scoresTensor, centernessTensor, vertexTensor, occlusionsTensor};
            // auto result = postprocesser.run(featuremaps);

            std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
            timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
        }

        gLogInfo << "Mnasneta1FCOS FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;
    });





    std::thread unetthread([&](){
        const float PRE_DIV = 1.0f / 255;
        double timesum = 0.0;
        for (int t = 0; t < EVALUATE_TIMES; ++t)
        {

            cv::Mat imnet = cv::imread(cv::format(IMPATH.c_str(), (t % 10) + 1));
            const uint8_t *shiftsrc = imnet.data + (imnet.rows - NETH) * NETW * 3; // pass upper area, which is mostly sky

            for (int i = 0; i < NET_PLANESIZE; ++i)
            {
                intensorPtrR2[i] = shiftsrc[3 * i + 2] * PRE_DIV;
                intensorPtrG2[i] = shiftsrc[3 * i + 1] * PRE_DIV;
                intensorPtrB2[i] = shiftsrc[3 * i + 0] * PRE_DIV;
            }

            std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


            buffers_UNET.copyInputToDevice();
            context_UNET->execute(1, buffers_UNET.getDeviceBindings().data());
            buffers_UNET.copyOutputToHost();


            std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
            timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());

        }

        gLogInfo << "MobilenetV2Unet FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;
    });


    fcosthread.join();
    unetthread.join();





    nvcaffeparser1::shutdownProtobufLibrary();

    return 0;
}