#include <signal.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rxcpp/rx.hpp>
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

#include "log_stream.hpp"
#include "nv/run_model.hpp"
#include "postprocess_fcos.hpp"


static auto LOG = spdlog::stdout_color_mt("MAIN");





bool isTimeToStop;
std::mutex imageQueueMutex;
std::queue<cv::Mat> imageQueue;

void *inputHost;
void *inputDevice;

void *clsScoreHost;
void *clsScoreDevice;
void *centernessHost;
void *centernessDevice;
void *bboxPredHost;
void *bboxPredDevice;
void *segHost;
void *segDevice;

void *encOutL3Device;
void *encOutL3DeviceClone;
void *encOutL2Device;
void *encOutL2DeviceClone;
void *encOutL1Device;
void *encOutL1DeviceClone;
void *downAvg4Device;
void *downAvg4DeviceClone;





inline void sighandler(int signo)
{
    isTimeToStop = true;
}


void deleteBuffers(void)
{
    // inputs
    free(inputHost);
    cudaFree(inputDevice);

    // outputs
    free(clsScoreHost);
    cudaFree(clsScoreDevice);
    free(centernessHost);
    cudaFree(centernessDevice);
    free(bboxPredHost);
    cudaFree(bboxPredDevice);
    free(segHost);
    cudaFree(segDevice);

    // middle layers
    cudaFree(encOutL3Device);
    cudaFree(encOutL2Device);
    cudaFree(encOutL1Device);
    cudaFree(downAvg4Device);
    cudaFree(encOutL3DeviceClone);
    cudaFree(encOutL2DeviceClone);
    cudaFree(encOutL1DeviceClone);
    cudaFree(downAvg4DeviceClone);
}


void allocBuffers(void)
{
    // inputs
    inputHost = malloc(3 * 480 * 640 * 4);
    cudaMalloc(&inputDevice, 3 * 480 * 640 * 4);

    // outputs
    clsScoreHost = malloc(3 * 60 * 80 * 4);
    cudaMalloc(&clsScoreDevice, 3 * 60 * 80 * 4);
    centernessHost = malloc(1 * 60 * 80 * 4);
    cudaMalloc(&centernessDevice, 1 * 60 * 80 * 4);
    bboxPredHost = malloc(4 * 60 * 80 * 4);
    cudaMalloc(&bboxPredDevice, 4 * 60 * 80 * 4);
    segHost = malloc(5 * 480 * 640 * 4);
    cudaMalloc(&segDevice, 5 * 480 * 640 * 4);

    // middle layers
    cudaMalloc(&encOutL3Device, 128 * 60 * 80 * 4);
    cudaMalloc(&encOutL2Device, 64 * 120 * 160 * 4);
    cudaMalloc(&encOutL1Device, 32 * 240 * 320 * 4);
    cudaMalloc(&downAvg4Device, 3 * 30 * 40 * 4);

    cudaMalloc(&encOutL3DeviceClone, 128 * 60 * 80 * 4);
    cudaMalloc(&encOutL2DeviceClone, 64 * 120 * 160 * 4);
    cudaMalloc(&encOutL1DeviceClone, 32 * 240 * 320 * 4);
    cudaMalloc(&downAvg4DeviceClone, 3 * 30 * 40 * 4);
}


rxcpp::observable<int32_t> imageObservable(cv::VideoCapture& cap)
{
    return rxcpp::observable<>::create<int32_t>
    (
        [&cap](rxcpp::subscriber<int> s)
        {
            while (isTimeToStop != true)
            {
                cv::Mat frame;
                cap >> frame;
                if (!frame.empty())
                {
                    std::lock_guard<std::mutex> lock(imageQueueMutex);
                    imageQueue.push(frame.clone());
                }
            }
            s.on_completed();
        }
    ).subscribe_on(rxcpp::synchronize_new_thread());
}


void plotBboxResult(cv::Mat &im, std::vector<std::vector<KeyPoint>> &result)
{
    const int numClass = result.size();
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        std::vector<KeyPoint> &shapes = result[clsIdx];
        const int numShape = shapes.size();
        for (int sIdx = 0; sIdx < numShape; ++sIdx)
        {
            cv::Point vertices[2];
            vertices[0].x = shapes[sIdx].vertexTL.x * im.cols;
            vertices[0].y = shapes[sIdx].vertexTL.y * im.rows;
            vertices[1].x = shapes[sIdx].vertexBR.x * im.cols;
            vertices[1].y = shapes[sIdx].vertexBR.y * im.rows;

            cv::rectangle(im, vertices[0], vertices[1], cv::Scalar(0, 0, 255), 1, 16);
        }
    }
}



int main (int ac, char *av[])
{
    if (ac != 2)
    {
        SLOG_ERROR << av[0] << " [config_file.json]" << std::endl;
        return 1;
    }

    nlohmann::json config;
    std::ifstream fin;
    fin.open(av[1]);
    fin >> config;
    fin.close();

    PostprocessFCOS postprocesser(config["model"]);

    allocBuffers();
    gallopwave::NVLogger logger;
    initLibNvInferPlugins(&logger, "");
    std::ifstream engineFile1("/home/gw/NN-inference-edge/data/espnet_0716_det_fp16.engine", std::ios::binary);
    std::vector<char> engineFileStream1(std::istreambuf_iterator<char>(engineFile1), {});
    auto runtime1 = gallopwave::NVUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine1 = gallopwave::NVUniquePtr<nvinfer1::ICudaEngine>(runtime1->deserializeCudaEngine(engineFileStream1.data(), engineFileStream1.size(), nullptr));
    auto context1 = gallopwave::NVUniquePtr<nvinfer1::IExecutionContext>(engine1->createExecutionContext());
    engineFile1.close();
    cudaStream_t stream1;
    cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, -1);
    std::vector<void *> devbinds1;
    devbinds1.push_back(inputDevice);
    devbinds1.push_back(downAvg4Device);
    devbinds1.push_back(encOutL1Device);
    devbinds1.push_back(encOutL2Device);
    devbinds1.push_back(encOutL3Device);
    devbinds1.push_back(clsScoreDevice);
    devbinds1.push_back(centernessDevice);
    devbinds1.push_back(bboxPredDevice);



    rxcpp::composite_subscription subscriptions;
    cv::VideoCapture cap;
    cap.open("/home/gw/g6g7_gaia2.mp4");
    
    isTimeToStop = false;
    signal(SIGINT, sighandler);
    
    imageObservable(cap).subscribe().add(subscriptions);



    float* intensorPtrR = static_cast<float*>(inputHost);
    float* intensorPtrG = intensorPtrR + 640 * 480;
    float* intensorPtrB = intensorPtrG + 640 * 480;



    int idxidx = 0;
    while (isTimeToStop != true)
    {
        std::lock_guard<std::mutex> lock(imageQueueMutex);
        if (!imageQueue.empty())
        {
            cv::Mat im = imageQueue.front();
            imageQueue.pop();


            cv::Mat imnet;
            cv::resize(im, imnet, cv::Size(640, 480));
            for (int i = 0; i < 307200; ++i)
            {
                intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - 0.4902f) / 0.2799f;
                intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - 0.4940f) / 0.2853f;
                intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - 0.4946f) / 0.2711f;
            }
            cudaMemcpyAsync(inputDevice, inputHost, 3 * 480 * 640 * 4, cudaMemcpyHostToDevice, stream1);
            context1->enqueue(1, devbinds1.data(), stream1, nullptr);
            cudaMemcpyAsync(clsScoreHost, clsScoreDevice, 3 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(centernessHost, centernessDevice, 1 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(bboxPredHost, bboxPredDevice, 4 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaStreamSynchronize(stream1); // Wait for the work in the stream to complete


            const float* scoresTensor = static_cast<const float*>(clsScoreHost);
            const float* vertexTensor = static_cast<const float*>(bboxPredHost);
            const float* centernessTensor = static_cast<const float*>(centernessHost);
            std::vector<const float *> featuremaps {scoresTensor, centernessTensor, vertexTensor};
            auto fcosResult = postprocesser.run(featuremaps);
            plotBboxResult(imnet, fcosResult);

            cv::imwrite("result" + std::to_string(idxidx) + ".jpg", imnet);
            ++idxidx;
        }
    }

    cudaStreamDestroy(stream1);

    subscriptions.unsubscribe();


    deleteBuffers();

    std::cout << "fin" << std::endl;

    return 0;
}