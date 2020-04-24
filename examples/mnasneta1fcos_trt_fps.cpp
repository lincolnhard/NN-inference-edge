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


void plotShapes(cv::Mat &im, std::vector<std::vector<KeyPoint>> &result, std::vector<int> num_vertex_byclass)
{
    const int numClass = result.size();
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        std::vector<KeyPoint> &shapes = result[clsIdx];
        const int numShape = shapes.size();
        const int numTotalVertex = num_vertex_byclass[clsIdx];
        for (int sIdx = 0; sIdx < numShape; ++sIdx)
        {
            cv::Point vertices[1][6];
            for (int i = 0; i < numTotalVertex; ++i)
            {
                vertices[0][i].x = shapes[sIdx].vertex[i].x * im.cols;
                vertices[0][i].y = shapes[sIdx].vertex[i].y * im.rows;
            }
            const cv::Point* vtsptr[1] = {vertices[0]};
            int npt[] = {numTotalVertex};
            cv::polylines(im, vtsptr, npt, 1, 1, cv::Scalar(0, 0, 255), 1, 16);
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
    const std::string IMPATH = config["evaluate"]["images_for_fps"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();

    PostprocessFCOS postprocesser(config["model"]);



    std::ifstream engineFile(TRT_ENGINE_PATH, std::ios::binary);
    std::vector<char> engineFileStream(std::istreambuf_iterator<char>(engineFile), {});

    nvinfer1::IRuntime* runtime = createInferRuntime(gLogger);
    std::shared_ptr<nvinfer1::ICudaEngine> engine =
        std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineFileStream.data(), engineFileStream.size(), nullptr), samplesCommon::InferDeleter());
    runtime->destroy();

    std::cout << engine->getBindingDimensions(0).d[0] << std::endl;
    std::cout << engine->getBindingDimensions(0).d[1] << std::endl;
    std::cout << engine->getBindingDimensions(0).d[2] << std::endl;
    

    samplesCommon::BufferManager buffers(engine, 1);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    float* intensorPtrR = static_cast<float*>(buffers.getHostBuffer("data"));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;


    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        cv::Mat imnet = cv::imread(cv::format(IMPATH.c_str(), (t % 10) + 1));
        for (int i = 0; i < NET_PLANESIZE; ++i)
        {
            intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
            intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
            intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
        }



        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        buffers.copyInputToDevice(); // Memcpy from host input buffers to device input buffers
        context->execute(1, buffers.getDeviceBindings().data());
        buffers.copyOutputToHost(); // Memcpy from device output buffers to host output buffers

        const float* scoresTensor = static_cast<const float*>(buffers.getHostBuffer("scoremap_perm"));
        const float* centernessTensor = static_cast<const float*>(buffers.getHostBuffer("centernessmap_perm"));
        const float* vertexTensor = static_cast<const float*>(buffers.getHostBuffer("regressionmap_perm"));
        const float* occlusionsTensor = static_cast<const float*>(buffers.getHostBuffer("occlusionmap_perm"));
        std::vector<const float *> featuremaps {scoresTensor, centernessTensor, vertexTensor, occlusionsTensor};
        auto result = postprocesser.run(featuremaps);


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());

        plotShapes(imnet, result, NUM_VERTEX_BYCLASS);
        cv::imwrite(cv::format("result%d.jpg", (t % 10) + 1), imnet);
    }

    // gLogInfo << "Mnasneta1FCOS FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;
    // gLogInfo << "Time consumed: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count() / 10.0 << std::endl;

    nvcaffeparser1::shutdownProtobufLibrary();

    return 0;
}