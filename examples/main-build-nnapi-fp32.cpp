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

    builder.getSdkVersion();
    builder.getDevices();

    // NNAPI default data layout NHWC
    const uint32_t NET_WIDTH = 512;
    const uint32_t NET_HEIGHT = 512;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // input
    float *indataptr = new float[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 0.5f);

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
    builder.addTensor("conv1_weight", {32, 3, 3, 3}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv1_bias", {32}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv1", "data", "conv1_weight", "conv1_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, false, ANEURALNETWORKS_FUSED_RELU6, "conv1_out");

    builder.addTensor("conv2_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv2_bias", {32}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv2", "conv1_out", "conv2_weight", "conv2_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv2_out");

    builder.addTensor("conv3_weight", {16, 1, 1, 32}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv3_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv3", "conv2_out", "conv3_weight", "conv3_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv3_out");

    builder.addTensor("conv4_weight", {96, 1, 1, 16}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv4_bias", {96}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv4", "conv3_out", "conv4_weight", "conv4_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv4_out");

    builder.addTensor("conv5_weight", {1, 3, 3, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv5_bias", {96}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv5", "conv4_out", "conv5_weight", "conv5_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv5_out");

    builder.addTensor("conv6_weight", {24, 1, 1, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv6_bias", {24}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv6", "conv5_out", "conv6_weight", "conv6_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv6_out");

    builder.addTensor("conv7_weight", {144, 1, 1, 24}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv7_bias", {144}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv7", "conv6_out", "conv7_weight", "conv7_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv7_out");

    builder.addTensor("conv8_weight", {1, 3, 3, 144}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv8_bias", {144}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv8", "conv7_out", "conv8_weight", "conv8_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv8_out");

    builder.addTensor("conv9_weight", {24, 1, 1, 144}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv9_bias", {24}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv9", "conv8_out", "conv9_weight", "conv9_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv9_out");

    builder.eltwiseAdd("add9", "conv6_out", "conv9_out", ANEURALNETWORKS_FUSED_NONE,
                        "add9_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv10_weight", {144, 1, 1, 24}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv10_bias", {144}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv10", "add9_out", "conv10_weight", "conv10_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv10_out");

    builder.addTensor("conv11_weight", {1, 3, 3, 144}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv11_bias", {144}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv11", "conv10_out", "conv11_weight", "conv11_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv11_out");

    builder.addTensor("conv12_weight", {32, 1, 1, 144}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv12_bias", {32}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv12", "conv11_out", "conv12_weight", "conv12_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv12_out");

    builder.addTensor("conv13_weight", {192, 1, 1, 32}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv13_bias", {192}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv13", "conv12_out", "conv13_weight", "conv13_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv13_out");

    builder.addTensor("conv14_weight", {1, 3, 3, 192}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv14_bias", {192}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv14", "conv13_out", "conv14_weight", "conv14_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv14_out");

    builder.addTensor("conv15_weight", {32, 1, 1, 192}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv15_bias", {32}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv15", "conv14_out", "conv15_weight", "conv15_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv15_out");

    builder.eltwiseAdd("add15", "conv12_out", "conv15_out", ANEURALNETWORKS_FUSED_NONE,
                        "add15_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv16_weight", {192, 1, 1, 32}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv16_bias", {192}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv16", "add15_out", "conv16_weight", "conv16_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv16_out");

    builder.addTensor("conv17_weight", {1, 3, 3, 192}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv17_bias", {192}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv17", "conv16_out", "conv17_weight", "conv17_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv17_out");

    builder.addTensor("conv18_weight", {32, 1, 1, 192}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv18_bias", {32}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv18", "conv17_out", "conv18_weight", "conv18_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv18_out");

    builder.eltwiseAdd("add18", "add15_out", "conv18_out", ANEURALNETWORKS_FUSED_NONE,
                        "add18_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv19_weight", {192, 1, 1, 32}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv19_bias", {192}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv19", "add18_out", "conv19_weight", "conv19_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv19_out");

    builder.addTensor("conv20_weight", {1, 3, 3, 192}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv20_bias", {192}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv20", "conv19_out", "conv20_weight", "conv20_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv20_out");

    builder.addTensor("conv21_weight", {64, 1, 1, 192}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv21_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv21", "conv20_out", "conv21_weight", "conv21_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv21_out");

    builder.addTensor("conv22_weight", {384, 1, 1, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv22_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv22", "conv21_out", "conv22_weight", "conv22_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv22_out");

    builder.addTensor("conv23_weight", {1, 3, 3, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv23_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv23", "conv22_out", "conv23_weight", "conv23_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv23_out");

    builder.addTensor("conv24_weight", {64, 1, 1, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv24_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv24", "conv23_out", "conv24_weight", "conv24_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv24_out");

    builder.eltwiseAdd("add24", "conv21_out", "conv24_out", ANEURALNETWORKS_FUSED_NONE,
                        "add24_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv25_weight", {384, 1, 1, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv25_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv25", "add24_out", "conv25_weight", "conv25_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv25_out");

    builder.addTensor("conv26_weight", {1, 3, 3, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv26_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv26", "conv25_out", "conv26_weight", "conv26_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv26_out");

    builder.addTensor("conv27_weight", {64, 1, 1, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv27_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv27", "conv26_out", "conv27_weight", "conv27_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv27_out");

    builder.eltwiseAdd("add27", "add24_out", "conv27_out", ANEURALNETWORKS_FUSED_NONE,
                        "add27_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv28_weight", {384, 1, 1, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv28_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv28", "add27_out", "conv28_weight", "conv28_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv28_out");

    builder.addTensor("conv29_weight", {1, 3, 3, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv29_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv29", "conv28_out", "conv29_weight", "conv29_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv29_out");

    builder.addTensor("conv30_weight", {64, 1, 1, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv30_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv30", "conv29_out", "conv30_weight", "conv30_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv30_out");

    builder.eltwiseAdd("add30", "add27_out", "conv30_out", ANEURALNETWORKS_FUSED_NONE,
                        "add30_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv31_weight", {384, 1, 1, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv31_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv31", "add30_out", "conv31_weight", "conv31_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv31_out");

    builder.addTensor("conv32_weight", {1, 3, 3, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv32_bias", {384}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv32", "conv31_out", "conv32_weight", "conv32_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv32_out");

    builder.addTensor("conv33_weight", {96, 1, 1, 384}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv33_bias", {96}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv33", "conv32_out", "conv33_weight", "conv33_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv33_out");

    builder.addTensor("conv34_weight", {576, 1, 1, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv34_bias", {576}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv34", "conv33_out", "conv34_weight", "conv34_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv34_out");

    builder.addTensor("conv35_weight", {1, 3, 3, 576}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv35_bias", {576}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv35", "conv34_out", "conv35_weight", "conv35_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv35_out");

    builder.addTensor("conv36_weight", {96, 1, 1, 576}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv36_bias", {96}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv36", "conv35_out", "conv36_weight", "conv36_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv36_out");

    builder.eltwiseAdd("add36", "conv33_out", "conv36_out", ANEURALNETWORKS_FUSED_NONE,
                        "add36_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv37_weight", {576, 1, 1, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv37_bias", {576}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv37", "add36_out", "conv37_weight", "conv37_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv37_out");

    builder.addTensor("conv38_weight", {1, 3, 3, 576}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv38_bias", {576}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv38", "conv37_out", "conv38_weight", "conv38_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv38_out");

    builder.addTensor("conv39_weight", {96, 1, 1, 576}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv39_bias", {96}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv39", "conv38_out", "conv39_weight", "conv39_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv39_out");

    builder.eltwiseAdd("add39", "add36_out", "conv39_out", ANEURALNETWORKS_FUSED_NONE,
                        "add39_out", ANEURALNETWORKS_TENSOR_FLOAT32);


    ////////////
    // add39_out
    ////////////

    builder.addTensor("conv40_weight", {576, 1, 1, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv40_bias", {576}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv40", "add39_out", "conv40_weight", "conv40_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv40_out");

    builder.addTensor("conv41_weight", {1, 3, 3, 576}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv41_bias", {576}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv41", "conv40_out", "conv41_weight", "conv41_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv41_out");

    builder.addTensor("conv42_weight", {160, 1, 1, 576}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv42_bias", {160}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv42", "conv41_out", "conv42_weight", "conv42_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv42_out");

    builder.addTensor("conv43_weight", {960, 1, 1, 160}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv43_bias", {960}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv43", "conv42_out", "conv43_weight", "conv43_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv43_out");

    builder.addTensor("conv44_weight", {1, 3, 3, 960}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv44_bias", {960}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv44", "conv43_out", "conv44_weight", "conv44_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv44_out");

    builder.addTensor("conv45_weight", {160, 1, 1, 960}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv45_bias", {160}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv45", "conv44_out", "conv45_weight", "conv45_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv45_out");

    builder.eltwiseAdd("add45", "conv42_out", "conv45_out", ANEURALNETWORKS_FUSED_NONE,
                        "add45_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv46_weight", {960, 1, 1, 160}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv46_bias", {960}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv46", "add45_out", "conv46_weight", "conv46_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv46_out");

    builder.addTensor("conv47_weight", {1, 3, 3, 960}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv47_bias", {960}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv47", "conv46_out", "conv47_weight", "conv47_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv47_out");

    builder.addTensor("conv48_weight", {160, 1, 1, 960}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv48_bias", {160}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv48", "conv47_out", "conv48_weight", "conv48_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv48_out");

    builder.eltwiseAdd("add48", "add45_out", "conv48_out", ANEURALNETWORKS_FUSED_NONE,
                        "add48_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    builder.addTensor("conv49_weight", {960, 1, 1, 160}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv49_bias", {960}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv49", "add48_out", "conv49_weight", "conv49_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv49_out");

    builder.addTensor("conv50_weight", {1, 3, 3, 960}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv50_bias", {960}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv50", "conv49_out", "conv50_weight", "conv50_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv50_out");

    builder.addTensor("conv51_weight", {320, 1, 1, 960}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv51_bias", {320}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv51", "conv50_out", "conv51_weight", "conv51_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv51_out");

    builder.addTensor("conv52_weight", {1280, 1, 1, 320}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv52_bias", {1280}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv52", "conv51_out", "conv52_weight", "conv52_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv52_out");

    /////////////
    // conv52_out
    /////////////


    builder.addTensor("conv53_weight", {256, 1, 1, 1280}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv53_bias", {256}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv53", "conv52_out", "conv53_weight", "conv53_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv53_out");

    builder.addTensor("conv54_weight", {1, 3, 3, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv54_bias", {256}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv54", "conv53_out", "conv54_weight", "conv54_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv54_out");

    builder.addTensor("conv55_weight", {512, 1, 1, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv55_bias", {512}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv55", "conv54_out", "conv55_weight", "conv55_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv55_out");

    /////////////
    // conv55_out
    /////////////


    builder.addTensor("conv56_weight", {128, 1, 1, 512}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv56_bias", {128}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv56", "conv55_out", "conv56_weight", "conv56_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv56_out");

    builder.addTensor("conv57_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv57_bias", {128}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv57", "conv56_out", "conv57_weight", "conv57_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv57_out");

    builder.addTensor("conv58_weight", {256, 1, 1, 128}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv58_bias", {256}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv58", "conv57_out", "conv58_weight", "conv58_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv58_out");


    /////////////
    // conv58_out
    /////////////


    builder.addTensor("conv59_weight", {128, 1, 1, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv59_bias", {128}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv59", "conv58_out", "conv59_weight", "conv59_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv59_out");

    builder.addTensor("conv60_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv60_bias", {128}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv60", "conv59_out", "conv60_weight", "conv60_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv60_out");

    builder.addTensor("conv61_weight", {256, 1, 1, 128}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv61_bias", {256}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv61", "conv60_out", "conv61_weight", "conv61_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv61_out");


    /////////////
    // conv61_out
    /////////////

    builder.addTensor("conv62_weight", {64, 1, 1, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv62_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv62", "conv61_out", "conv62_weight", "conv62_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv62_out");

    builder.addTensor("conv63_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv63_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv63", "conv62_out", "conv63_weight", "conv63_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv63_out");

    builder.addTensor("conv64_weight", {64, 1, 1, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv64_bias", {64}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv64", "conv63_out", "conv64_weight", "conv64_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv64_out");

    /////////////
    // conv64_out
    /////////////
#if 0
    builder.addTensor("conv65_weight", {84, 3, 3, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv65_bias", {84}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv65", "add39_out", "conv65_weight", "conv65_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv65_out");

    builder.addTensor("conv66_weight", {126, 3, 3, 1280}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv66_bias", {126}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv66", "conv52_out", "conv66_weight", "conv66_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv66_out");

    builder.addTensor("conv67_weight", {126, 3, 3, 512}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv67_bias", {126}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv67", "conv55_out", "conv67_weight", "conv67_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv67_out");

    builder.addTensor("conv68_weight", {126, 3, 3, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv68_bias", {126}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv68", "conv58_out", "conv68_weight", "conv68_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv68_out");

    builder.addTensor("conv69_weight", {126, 3, 3, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv69_bias", {126}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv69", "conv61_out", "conv69_weight", "conv69_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv69_out");

    builder.addTensor("conv70_weight", {84, 3, 3, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv70_bias", {84}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv70", "conv64_out", "conv70_weight", "conv70_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv70_out");

    ///////////

    builder.addTensor("conv71_weight", {16, 3, 3, 96}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv71_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv71", "add39_out", "conv71_weight", "conv71_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv71_out");

    builder.addTensor("conv72_weight", {24, 3, 3, 1280}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv72_bias", {24}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv72", "conv52_out", "conv72_weight", "conv72_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv72_out");

    builder.addTensor("conv73_weight", {24, 3, 3, 512}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv73_bias", {24}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv73", "conv55_out", "conv73_weight", "conv73_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv73_out");

    builder.addTensor("conv74_weight", {24, 3, 3, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv74_bias", {24}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv74", "conv58_out", "conv74_weight", "conv74_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv74_out");

    builder.addTensor("conv75_weight", {24, 3, 3, 256}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv75_bias", {24}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv75", "conv61_out", "conv75_weight", "conv75_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv75_out");

    builder.addTensor("conv76_weight", {16, 3, 3, 64}, ANEURALNETWORKS_TENSOR_FLOAT32, weightptr);
    builder.addTensor("conv76_bias", {16}, ANEURALNETWORKS_TENSOR_FLOAT32, biasptr);
    builder.conv2d("conv76", "conv64_out", "conv76_weight", "conv76_bias", ANEURALNETWORKS_TENSOR_FLOAT32,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv76_out");
#endif



    // set input/output
    builder.setInputOps("data", indataptr, ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv65_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv66_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv67_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv68_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv69_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv70_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv71_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv72_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv73_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv74_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv75_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    // builder.setOutputOps("conv76_out", ANEURALNETWORKS_TENSOR_FLOAT32);
    builder.setOutputOps("conv64_out", ANEURALNETWORKS_TENSOR_FLOAT32);

    // compile
    builder.compile(deviceIndex);

    // execute
    double timesum = 0.0;
    const int EVALUATE_TIMES = 100;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

        builder.execute();

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }
    SLOG_INFO << "FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;



    // validate
    std::vector<void *> results = builder.getOutput();

    for (int32_t idx = 0; idx < 10; ++idx)
    {
        float *result0 = reinterpret_cast<float *>(results[0]);
        SLOG_INFO << result0[idx] << std::endl;
    }


    delete [] indataptr;
    delete [] weightptr;
    delete [] biasptr;

    SLOG_INFO << "fine" << std::endl;
    return 0;
}