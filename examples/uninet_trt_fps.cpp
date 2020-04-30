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


#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "trt_buffers.h"


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


int main(int ac, char *av[])
{
    if (ac != 2)
    {
        LOG(ERROR) << av[0] << " [config_file.json]" << std::endl;
        return 1;
    }

    TrtLogger trtlogger;
    cudaSetDevice(0); // TX2 has only one device
    initLibNvInferPlugins(&trtlogger, "");

    nlohmann::json config;
    std::ifstream fin;
    fin.open(av[1]);
    fin >> config;
    fin.close();



    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;

    std::ifstream engineFile(TRT_ENGINE_PATH, std::ios::binary);
    std::vector<char> engineFileStream(std::istreambuf_iterator<char>(engineFile), {});

    std::shared_ptr<nvinfer1::IRuntime> runtime =
        std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger), InferDeleter());
    std::shared_ptr<nvinfer1::ICudaEngine> engine =
        std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineFileStream.data(), engineFileStream.size(), nullptr), InferDeleter());
    std::shared_ptr<nvinfer1::IExecutionContext> context =
        std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext(), InferDeleter());

    LOG(INFO) << engine->getNbBindings() << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        LOG(INFO) << engine->getBindingName(i) << " " << engine->bindingIsInput(i) << std::endl;
        nvinfer1::Dims outDim = engine->getBindingDimensions(i);
        for (int j = 0; j < outDim.nbDims; ++j)
        {
            LOG(INFO) << outDim.d[j] << std::endl;
        }
    }

    BufferManager buffers(engine, 1);
    float* intensorPtrR = static_cast<float*>(buffers.getHostBuffer("transpose"));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;

    cv::Mat im = cv::imread(IMPATH);
    cv::Mat imnet;
    cv::resize(im, imnet, cv::Size(NETW, NETH));
    for (int i = 0; i < NET_PLANESIZE; ++i)
    {
        intensorPtrR[i] = imnet.data[3 * i + 2] / 255.0f;
        intensorPtrG[i] = imnet.data[3 * i + 1] / 255.0f;
        intensorPtrB[i] = imnet.data[3 * i + 0] / 255.0f;
    }

    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        buffers.copyInputToDevice(); // Memcpy from host input buffers to device input buffers
        LOG(INFO) << "Memcpy from host input buffers to device input buffers" << std::endl;

        context->execute(1, buffers.getDeviceBindings().data());
        LOG(INFO) << "Synchronos execute" << std::endl;

        buffers.copyOutputToHost(); // Memcpy from device output buffers to host output buffers
        LOG(INFO) << "Memcpy from device output buffers to host output buffers" << std::endl;


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }

    LOG(INFO) << "Uninet FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;

    return 0;
}