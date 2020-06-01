#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "nv/run_model.hpp"
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>




static auto LOG = spdlog::stdout_color_mt("MAIN");

void *inputDevice;
void *clsScoreDevice;
void *centernessDevice;
void *bboxPredDevice;
void *l3projDevice;
void *l2projDevice;
void *l1projDevice;
void *l40Device;
void *segDevice;


void *inputHost;
void *clsScoreHost;
void *centernessHost;
void *bboxPredHost;
void *segHost;

void *l3projDeviceClone;
void *l2projDeviceClone;
void *l1projDeviceClone;
void *l40DeviceClone;

std::mutex thmutex;
bool isTimeToStop = false;


int main(int ac, char *av[])
{
    auto ret = cudaMalloc(&inputDevice, 3 * 480 * 640 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&clsScoreDevice, 76 * 60 * 80 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&centernessDevice, 1 * 60 * 80 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&bboxPredDevice, 4 * 60 * 80 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l3projDevice, 64 * 60 * 80 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l2projDevice, 48 * 120 * 160 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l1projDevice, 32 * 240 * 320 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l40Device, 256 * 30 * 40 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&segDevice, 13 * 240 * 320 * 4);
    assert(ret == cudaSuccess);


    inputHost = malloc(3 * 480 * 640 * 4);
    assert(inputHost != nullptr);
    clsScoreHost = malloc(76 * 60 * 80 * 4);
    assert(clsScoreHost != nullptr);
    centernessHost = malloc(1 * 60 * 80 * 4);
    assert(centernessHost != nullptr);
    bboxPredHost = malloc(4 * 60 * 80 * 4);
    assert(bboxPredHost != nullptr);
    segHost = malloc(13 * 240 * 320 * 4);
    assert(segHost != nullptr);


    ret = cudaMalloc(&l3projDeviceClone, 64 * 60 * 80 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l2projDeviceClone, 48 * 120 * 160 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l1projDeviceClone, 32 * 240 * 320 * 4);
    assert(ret == cudaSuccess);
    ret = cudaMalloc(&l40DeviceClone, 256 * 30 * 40 * 4);
    assert(ret == cudaSuccess);



    // int priorityHigh = 0;
    // int priorityLow = 0;
    // cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
    // SLOG_INFO << "priorityHigh: " << priorityHigh << std::endl;
    // SLOG_INFO << "priorityLow: " << priorityLow << std::endl;



    std::thread detTh = std::thread([](){

        gallopwave::NVLogger logger;

        initLibNvInferPlugins(&logger, "");

        std::ifstream engineFile1("/home/gw/Documents/NN-inference-edge/data/espnetv2_part1_det_fp16.engine", std::ios::binary);
        std::vector<char> engineFileStream1(std::istreambuf_iterator<char>(engineFile1), {});
        auto runtime1 = gallopwave::NVUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        auto engine1 = gallopwave::NVUniquePtr<nvinfer1::ICudaEngine>(runtime1->deserializeCudaEngine(engineFileStream1.data(), engineFileStream1.size(), nullptr));
        auto context1 = gallopwave::NVUniquePtr<nvinfer1::IExecutionContext>(engine1->createExecutionContext());
        engineFile1.close();

        cudaStream_t stream1;
        cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, -1);


        std::vector<void *> devbinds1;
        devbinds1.push_back(inputDevice);
        devbinds1.push_back(clsScoreDevice);
        devbinds1.push_back(centernessDevice);
        devbinds1.push_back(bboxPredDevice);
        devbinds1.push_back(l3projDevice);
        devbinds1.push_back(l2projDevice);
        devbinds1.push_back(l1projDevice);
        devbinds1.push_back(l40Device);
        
        double timesum = 0.0;
        for (int i = 0; i < 200; ++i)
        {
            std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

            cudaMemcpyAsync(inputDevice, inputHost, 3 * 480 * 640 * 4, cudaMemcpyHostToDevice, stream1);
            context1->enqueue(1, devbinds1.data(), stream1, nullptr);
            cudaMemcpyAsync(clsScoreHost, clsScoreDevice, 76 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(centernessHost, centernessDevice, 1 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(bboxPredHost, bboxPredDevice, 4 * 60 * 80 * 4, cudaMemcpyDeviceToHost, stream1);

            {
            std::lock_guard<std::mutex> lock(thmutex);
            cudaMemcpyAsync(l3projDeviceClone, l3projDevice, 64 * 60 * 80 * 4, cudaMemcpyDeviceToDevice, stream1);
            cudaMemcpyAsync(l2projDeviceClone, l2projDevice, 48 * 120 * 160 * 4, cudaMemcpyDeviceToDevice, stream1);
            cudaMemcpyAsync(l1projDeviceClone, l1projDevice, 32 * 240 * 320 * 4, cudaMemcpyDeviceToDevice, stream1);
            cudaMemcpyAsync(l40DeviceClone, l40Device, 256 * 30 * 40 * 4, cudaMemcpyDeviceToDevice, stream1);
            }

            cudaStreamSynchronize(stream1); // Wait for the work in the stream to complete

            std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
            timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
        }
        cudaStreamDestroy(stream1);
        SLOG_INFO << "DET FPS: " << 1.0 / (timesum / 200 / 1000.0) << std::endl;

        {
            std::lock_guard<std::mutex> lock(thmutex);
            isTimeToStop = true;
        }
    });


    std::thread segTh = std::thread([](){

        gallopwave::NVLogger logger;

        initLibNvInferPlugins(&logger, "");

        std::ifstream engineFile2("/home/gw/Documents/NN-inference-edge/data/espnetv2_part2_seg_fp16.engine", std::ios::binary);
        std::vector<char> engineFileStream2(std::istreambuf_iterator<char>(engineFile2), {});
        auto runtime2 = gallopwave::NVUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        auto engine2 = gallopwave::NVUniquePtr<nvinfer1::ICudaEngine>(runtime2->deserializeCudaEngine(engineFileStream2.data(), engineFileStream2.size(), nullptr));
        auto context2 = gallopwave::NVUniquePtr<nvinfer1::IExecutionContext>(engine2->createExecutionContext());
        engineFile2.close();



        cudaStream_t stream2;
        cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, 0);

        std::vector<void *> devbinds2;
        devbinds2.push_back(l3projDeviceClone);
        devbinds2.push_back(l2projDeviceClone);
        devbinds2.push_back(l1projDeviceClone);
        devbinds2.push_back(l40DeviceClone);
        devbinds2.push_back(segDevice);


        int loopCount = 0;
        double timesum = 0.0;
        while (1)
        {
            std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

            context2->enqueue(1, devbinds2.data(), stream2, nullptr);
            cudaMemcpyAsync(segHost, segDevice, 13 * 240 * 320 * 4, cudaMemcpyDeviceToHost, stream2);
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


    cudaFree(inputDevice);
    cudaFree(clsScoreDevice);
    cudaFree(centernessDevice);
    cudaFree(bboxPredDevice);
    cudaFree(l3projDevice);
    cudaFree(l2projDevice);
    cudaFree(l1projDevice);
    cudaFree(l40Device);
    cudaFree(segDevice);

    free(inputHost);
    free(clsScoreHost);
    free(centernessHost);
    free(bboxPredHost);
    free(segHost);

    cudaFree(l3projDeviceClone);
    cudaFree(l2projDeviceClone);
    cudaFree(l1projDeviceClone);
    cudaFree(l40DeviceClone);

    return 0;
}