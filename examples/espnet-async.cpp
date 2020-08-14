#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <experimental/filesystem>
#include <signal.h>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "nv/run_model.hpp"
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>



static auto LOG = spdlog::stdout_color_mt("MAIN");

void *inputDevice;
void *clsScoreDevice;
void *centernessDevice;
void *bboxPredDevice;
void *encOutL3Device;
void *encOutL2Device;
void *encOutL1Device;
void *downAvg4Device;
void *segDevice;


void *inputHost;
void *clsScoreHost;
void *centernessHost;
void *bboxPredHost;
void *segHost;

void *encOutL3DeviceClone;
void *encOutL2DeviceClone;
void *encOutL1DeviceClone;
void *downAvg4DeviceClone;

std::mutex thmutex;
bool isTimeToStop = false;
cv::Mat im;

void sigintHandler(int sig)
{
    std::lock_guard<std::mutex> lock(thmutex);
    isTimeToStop = true;
}

int main(int ac, char *av[])
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

    // interupt handling
    signal(SIGINT, sigintHandler);

    // set cuda stream priority
    // int priorityHigh = 0;
    // int priorityLow = 0;
    // cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
    // SLOG_INFO << "priorityHigh: " << priorityHigh << std::endl;
    // SLOG_INFO << "priorityLow: " << priorityLow << std::endl;



#if 0
    std::thread detTh = std::thread([](){

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



        
        double timesum = 0.0;
        for (int i = 0; i < 1000; ++i)
        {
            std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

            cudaMemcpyAsync(inputDevice, inputHost, 3 * 480 * 640 * 4, cudaMemcpyHostToDevice, stream1);
            context1->enqueue(1, devbinds1.data(), stream1, nullptr);
            cudaMemcpyAsync(clsScoreHost, clsScoreDevice, 3 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(centernessHost, centernessDevice, 1 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(bboxPredHost, bboxPredDevice, 4 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);

            {
            std::lock_guard<std::mutex> lock(thmutex);
            cudaMemcpyAsync(downAvg4DeviceClone, downAvg4Device, 3 * 30 * 40 * 4, cudaMemcpyDeviceToDevice, stream1);
            cudaMemcpyAsync(encOutL1DeviceClone, encOutL1Device, 32 * 240 * 320 * 4, cudaMemcpyDeviceToDevice, stream1);
            cudaMemcpyAsync(encOutL2DeviceClone, encOutL2Device, 64 * 120 * 160 * 4, cudaMemcpyDeviceToDevice, stream1);
            cudaMemcpyAsync(encOutL3DeviceClone, encOutL3Device, 128 * 60 * 80 * 4, cudaMemcpyDeviceToDevice, stream1);
            }

            cudaStreamSynchronize(stream1); // Wait for the work in the stream to complete

            std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
            timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
        }
        cudaStreamDestroy(stream1);
        SLOG_INFO << "DET FPS: " << 1.0 / (timesum / 1000 / 1000.0) << std::endl;

        {
            std::lock_guard<std::mutex> lock(thmutex);
            isTimeToStop = true;
        }
    });


    std::thread segTh = std::thread([](){

        gallopwave::NVLogger logger;

        initLibNvInferPlugins(&logger, "");

        std::ifstream engineFile2("/home/gw/NN-inference-edge/data/espnet_0716_seg_fp16.engine", std::ios::binary);
        std::vector<char> engineFileStream2(std::istreambuf_iterator<char>(engineFile2), {});
        auto runtime2 = gallopwave::NVUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        auto engine2 = gallopwave::NVUniquePtr<nvinfer1::ICudaEngine>(runtime2->deserializeCudaEngine(engineFileStream2.data(), engineFileStream2.size(), nullptr));
        auto context2 = gallopwave::NVUniquePtr<nvinfer1::IExecutionContext>(engine2->createExecutionContext());
        engineFile2.close();



        cudaStream_t stream2;
        cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, 0);

        std::vector<void *> devbinds2;
        devbinds2.push_back(encOutL3DeviceClone);
        devbinds2.push_back(encOutL2DeviceClone);
        devbinds2.push_back(encOutL1DeviceClone);
        devbinds2.push_back(downAvg4DeviceClone);
        devbinds2.push_back(segDevice);


        int loopCount = 0;
        double timesum = 0.0;
        while (1)
        {
            std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

            context2->enqueue(1, devbinds2.data(), stream2, nullptr);
            cudaMemcpyAsync(segHost, segDevice, 5 * 240 * 320 * 4, cudaMemcpyDeviceToHost, stream2);
            cudaStreamSynchronize(stream2); // Wait for the work in the stream to complete

            std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
            timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
            loopCount += 1;

            {
            std::lock_guard<std::mutex> lock(thmutex);
            if (isTimeToStop == true)
            {
                break;
            }
            }
        }

        cudaStreamDestroy(stream2);
        SLOG_INFO << "SEG FPS: " << 1.0 / (timesum / loopCount / 1000.0) << std::endl;
    });



    detTh.join();
    segTh.join();
#endif


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

    return 0;
}