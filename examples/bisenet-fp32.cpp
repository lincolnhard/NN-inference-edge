#include <iostream>
#include "napi/builder.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"

#include <chrono>
#include <thread>



static auto LOG = spdlog::stdout_color_mt("MAIN");

int main(int ac, char *av[])
{
    int32_t deviceIndex = -1;
    if (ac == 1)
    {
        deviceIndex = -1;
    }
    else if (ac == 2)
    {
        deviceIndex = std::stoi(std::string(av[1]));
    }
    else
    {
        SLOG_ERROR << "Usage: " << av[0] << " [device ID] (0: GPU, 1: APU, 2: CPU, -1: Auto-select)" << std::endl;
        return 1;
    }

    gallopwave::ModelBuilder builder;

    // NNAPI default data layout NHWC
    const uint32_t NET_WIDTH = 1024;
    const uint32_t NET_HEIGHT = 512;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // inputs
    float *indataptr = new float[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 0.6f);

    // kernels
    const uint32_t DUMMY_N = 1280;
    const uint32_t DUMMY_C = 100;
    const uint32_t DUMMY_H = 10;
    const uint32_t DUMMY_W = 10;
    const uint32_t DUMMY_KERNEL_SIZE = DUMMY_N * DUMMY_H * DUMMY_W * DUMMY_C;
    float *weightptr = new float[DUMMY_KERNEL_SIZE];
    std::fill(weightptr, weightptr + DUMMY_KERNEL_SIZE, 0.5f);
    float *biasptr = new float[DUMMY_N];
    std::fill(biasptr, biasptr + DUMMY_N, 0.5f);


    // start to build
    builder.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS}, ANEURALNETWORKS_TENSOR_FLOAT32);
    builder.addTensor("conv1_weight", {8, 3, 3, 3}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv1_bias", {8}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv1", "data", "conv1_weight", "conv1_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, false, ANEURALNETWORKS_FUSED_RELU, "conv1_out");

    builder.maxpool("pool1", "conv1_out", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, 3, 3, ANEURALNETWORKS_FUSED_NONE, "pool1_out");

    builder.addTensor("conv2_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv2_bias", {8}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv2", "pool1_out", "conv2_weight", "conv2_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv2_out");

    builder.addTensor("conv3_weight", {16, 1, 1, 8}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv3_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv3", "conv2_out", "conv3_weight", "conv3_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv3_out");

    builder.addTensor("conv4_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv4_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv4", "conv3_out", "conv4_weight", "conv4_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv4_out");

    builder.addTensor("conv5_weight", {16, 1, 1, 16}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv5_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv5", "conv4_out", "conv5_weight", "conv5_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv5_out");

    builder.addTensor("conv6_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv6_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv6", "conv5_out", "conv6_weight", "conv6_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv6_out");

    builder.addTensor("conv7_weight", {64, 1, 1, 16}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv7_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv7", "conv6_out", "conv7_weight", "conv7_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv7_out");

    builder.addTensor("conv8_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv8_bias", {8}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv8", "pool1_out", "conv8_weight", "conv8_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv8_out");

    builder.addTensor("conv9_weight", {64, 1, 1, 8}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv9_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv9", "conv8_out", "conv9_weight", "conv9_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv9_out");

    builder.eltwise("add1", "conv7_out", "conv9_out", ANEURALNETWORKS_FUSED_RELU,
                        "add1_out", ANEURALNETWORKS_TENSOR_FLOAT32, ELTWISE_ADDITION);











    builder.setInputTensors("data", indataptr, ANEURALNETWORKS_TENSOR_FLOAT32);
    builder.setOutputTensors("add1_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.compile(deviceIndex);
    builder.execute();

    std::vector<void *> results = builder.getOutput();

    for (int32_t idx = 0; idx < 60; ++idx)
    {
        float *result0 = reinterpret_cast<float *>(results[0]);
        SLOG_INFO << result0[idx] << std::endl;
    }




    delete [] indataptr;
    delete [] weightptr;
    delete [] biasptr;


    return 0;
}