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
    const uint32_t NET_WIDTH = 1024;
    const uint32_t NET_HEIGHT = 512;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // inputs
    uint8_t *indataptr = new uint8_t[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 1);

    // kernels
    const uint32_t DUMMY_N = 1280;
    const uint32_t DUMMY_C = 100;
    const uint32_t DUMMY_H = 10;
    const uint32_t DUMMY_W = 10;
    const uint32_t DUMMY_KERNEL_SIZE = DUMMY_N * DUMMY_H * DUMMY_W * DUMMY_C;
    uint8_t *weightptr = new uint8_t[DUMMY_KERNEL_SIZE];
    std::fill(weightptr, weightptr + DUMMY_KERNEL_SIZE, 2);
    uint8_t *biasptr = new uint8_t[DUMMY_N];
    std::fill(biasptr, biasptr + DUMMY_N, 1);


    // start to build
    builder.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.addTensor("conv1_weight", {8, 3, 3, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv1_bias", {8}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv1", "data", "conv1_weight", "conv1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, false, ANEURALNETWORKS_FUSED_RELU, "conv1_out");

    builder.maxpool("pool1", "conv1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, 3, 3, ANEURALNETWORKS_FUSED_NONE, "pool1_out");

    builder.addTensor("conv2_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv2_bias", {8}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv2", "pool1_out", "conv2_weight", "conv2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv2_out");

    builder.addTensor("conv3_weight", {16, 1, 1, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv3_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv3", "conv2_out", "conv3_weight", "conv3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv3_out");

    builder.addTensor("conv4_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv4_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv4", "conv3_out", "conv4_weight", "conv4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv4_out");

    builder.addTensor("conv5_weight", {16, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv5_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv5", "conv4_out", "conv5_weight", "conv5_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv5_out");

    builder.addTensor("conv6_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv6_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv6", "conv5_out", "conv6_weight", "conv6_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv6_out");

    builder.addTensor("conv7_weight", {64, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv7_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv7", "conv6_out", "conv7_weight", "conv7_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv7_out");

    builder.addTensor("conv8_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv8_bias", {8}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv8", "pool1_out", "conv8_weight", "conv8_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv8_out");

    builder.addTensor("conv9_weight", {64, 1, 1, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv9_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv9", "conv8_out", "conv9_weight", "conv9_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv9_out");

    builder.eltwise("add9", "conv7_out", "conv9_out", ANEURALNETWORKS_FUSED_RELU,
                        "add9_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv10_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv10_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv10", "add9_out", "conv10_weight", "conv10_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv10_out");

    builder.addTensor("conv11_weight", {16, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv11_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv11", "conv10_out", "conv11_weight", "conv11_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv11_out");

    builder.addTensor("conv12_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv12_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv12", "conv11_out", "conv12_weight", "conv12_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv12_out");

    builder.addTensor("conv13_weight", {16, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv13_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv13", "conv12_out", "conv13_weight", "conv13_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv13_out");

    builder.addTensor("conv14_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv14_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv14", "conv13_out", "conv14_weight", "conv14_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv14_out");

    builder.addTensor("conv15_weight", {64, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv15_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv15", "conv14_out", "conv15_weight", "conv15_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv15_out");

    builder.eltwise("add15", "add9_out", "conv15_out", ANEURALNETWORKS_FUSED_RELU,
                        "add15_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv16_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv16_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv16", "add15_out", "conv16_weight", "conv16_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv16_out");

    builder.addTensor("conv17_weight", {16, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv17_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv17", "conv16_out", "conv17_weight", "conv17_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv17_out");

    builder.addTensor("conv18_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv18_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv18", "conv17_out", "conv18_weight", "conv18_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv18_out");

    builder.addTensor("conv19_weight", {16, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv19_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv19", "conv18_out", "conv19_weight", "conv19_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv19_out");

    builder.addTensor("conv20_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv20_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv20", "conv19_out", "conv20_weight", "conv20_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv20_out");

    builder.addTensor("conv21_weight", {64, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv21_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv21", "conv20_out", "conv21_weight", "conv21_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv21_out");

    builder.eltwise("add21", "add15_out", "conv21_out", ANEURALNETWORKS_FUSED_RELU,
                        "add21_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv22_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv22_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv22", "add21_out", "conv22_weight", "conv22_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv22_out");

    builder.addTensor("conv23_weight", {16, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv23_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv23", "conv22_out", "conv23_weight", "conv23_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv23_out");

    builder.addTensor("conv24_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv24_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv24", "conv23_out", "conv24_weight", "conv24_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv24_out");

    builder.addTensor("conv25_weight", {16, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv25_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv25", "conv24_out", "conv25_weight", "conv25_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv25_out");

    builder.addTensor("conv26_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv26_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv26", "conv25_out", "conv26_weight", "conv26_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv26_out");

    builder.addTensor("conv27_weight", {64, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv27_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv27", "conv26_out", "conv27_weight", "conv27_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv27_out");

    builder.eltwise("add27", "add21_out", "conv27_out", ANEURALNETWORKS_FUSED_RELU,
                        "add27_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv28_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv28_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv28", "add27_out", "conv28_weight", "conv28_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv28_out");

    builder.addTensor("conv29_weight", {32, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv29_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv29", "conv28_out", "conv29_weight", "conv29_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv29_out");

    builder.addTensor("conv30_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv30_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv30", "conv29_out", "conv30_weight", "conv30_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv30_out");

    builder.addTensor("conv31_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv31_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv31", "conv30_out", "conv31_weight", "conv31_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv31_out");

    builder.addTensor("conv32_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv32_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv32", "conv31_out", "conv32_weight", "conv32_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv32_out");

    builder.addTensor("conv33_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv33_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv33", "conv32_out", "conv33_weight", "conv33_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv33_out");

    builder.addTensor("conv34_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv34_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv34", "add27_out", "conv34_weight", "conv34_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv34_out");

    builder.addTensor("conv35_weight", {128, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv35_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv35", "conv34_out", "conv35_weight", "conv35_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv35_out");

    builder.eltwise("add35", "conv33_out", "conv35_out", ANEURALNETWORKS_FUSED_RELU,
                        "add35_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv36_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv36_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv36", "add35_out", "conv36_weight", "conv36_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv36_out");

    builder.addTensor("conv37_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv37_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv37", "conv36_out", "conv37_weight", "conv37_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv37_out");

    builder.addTensor("conv38_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv38_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv38", "conv37_out", "conv38_weight", "conv38_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv38_out");

    builder.addTensor("conv39_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv39_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv39", "conv38_out", "conv39_weight", "conv39_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv39_out");

    builder.addTensor("conv40_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv40_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv40", "conv39_out", "conv40_weight", "conv40_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv40_out");

    builder.addTensor("conv41_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv41_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv41", "conv40_out", "conv41_weight", "conv41_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv41_out");

    builder.eltwise("add41", "add35_out", "conv41_out", ANEURALNETWORKS_FUSED_RELU,
                        "add41_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv42_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv42_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv42", "add41_out", "conv42_weight", "conv42_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv42_out");

    builder.addTensor("conv43_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv43_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv43", "conv42_out", "conv43_weight", "conv43_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv43_out");

    builder.addTensor("conv44_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv44_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv44", "conv43_out", "conv44_weight", "conv44_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv44_out");

    builder.addTensor("conv45_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv45_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv45", "conv44_out", "conv45_weight", "conv45_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv45_out");

    builder.addTensor("conv46_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv46_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv46", "conv45_out", "conv46_weight", "conv46_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv46_out");

    builder.addTensor("conv47_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv47_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv47", "conv46_out", "conv47_weight", "conv47_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv47_out");

    builder.eltwise("add47", "add41_out", "conv47_out", ANEURALNETWORKS_FUSED_RELU,
                        "add47_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv48_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv48_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv48", "add47_out", "conv48_weight", "conv48_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv48_out");

    builder.addTensor("conv49_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv49_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv49", "conv48_out", "conv49_weight", "conv49_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv49_out");

    builder.addTensor("conv50_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv50_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv50", "conv49_out", "conv50_weight", "conv50_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv50_out");

    builder.addTensor("conv51_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv51_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv51", "conv50_out", "conv51_weight", "conv51_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv51_out");

    builder.addTensor("conv52_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv52_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv52", "conv51_out", "conv52_weight", "conv52_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv52_out");

    builder.addTensor("conv53_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv53_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv53", "conv52_out", "conv53_weight", "conv53_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv53_out");

    builder.eltwise("add53", "add47_out", "conv53_out", ANEURALNETWORKS_FUSED_RELU,
                        "add53_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv54_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv54_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv54", "add53_out", "conv54_weight", "conv54_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv54_out");

    builder.addTensor("conv55_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv55_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv55", "conv54_out", "conv55_weight", "conv55_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv55_out");

    builder.addTensor("conv56_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv56_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv56", "conv55_out", "conv56_weight", "conv56_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv56_out");

    builder.addTensor("conv57_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv57_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv57", "conv56_out", "conv57_weight", "conv57_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv57_out");

    builder.addTensor("conv58_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv58_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv58", "conv57_out", "conv58_weight", "conv58_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv58_out");

    builder.addTensor("conv59_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv59_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv59", "conv58_out", "conv59_weight", "conv59_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv59_out");

    builder.eltwise("add59", "add53_out", "conv59_out", ANEURALNETWORKS_FUSED_RELU,
                        "add59_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv60_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv60_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv60", "add59_out", "conv60_weight", "conv60_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv60_out");

    builder.addTensor("conv61_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv61_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv61", "conv60_out", "conv61_weight", "conv61_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv61_out");

    builder.addTensor("conv62_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv62_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv62", "conv61_out", "conv62_weight", "conv62_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv62_out");

    builder.addTensor("conv63_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv63_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv63", "conv62_out", "conv63_weight", "conv63_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv63_out");

    builder.addTensor("conv64_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv64_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv64", "conv63_out", "conv64_weight", "conv64_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv64_out");

    builder.addTensor("conv65_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv65_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv65", "conv64_out", "conv65_weight", "conv65_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv65_out");

    builder.eltwise("add65", "add59_out", "conv65_out", ANEURALNETWORKS_FUSED_RELU,
                        "add65_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv66_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv66_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv66", "add65_out", "conv66_weight", "conv66_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv66_out");

    builder.addTensor("conv67_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv67_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv67", "conv66_out", "conv67_weight", "conv67_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv67_out");

    builder.addTensor("conv68_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv68_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv68", "conv67_out", "conv68_weight", "conv68_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv68_out");

    builder.addTensor("conv69_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv69_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv69", "conv68_out", "conv69_weight", "conv69_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv69_out");

    builder.addTensor("conv70_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv70_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv70", "conv69_out", "conv70_weight", "conv70_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv70_out");

    builder.addTensor("conv71_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv71_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv71", "conv70_out", "conv71_weight", "conv71_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv71_out");

    builder.eltwise("add71", "add65_out", "conv71_out", ANEURALNETWORKS_FUSED_RELU,
                        "add71_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv72_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv72_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv72", "add71_out", "conv72_weight", "conv72_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv72_out");

    builder.addTensor("conv73_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv73_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv73", "conv72_out", "conv73_weight", "conv73_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv73_out");

    builder.addTensor("conv74_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv74_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv74", "conv73_out", "conv74_weight", "conv74_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv74_out");

    builder.addTensor("conv75_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv75_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv75", "conv74_out", "conv75_weight", "conv75_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv75_out");

    builder.addTensor("conv76_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv76_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv76", "conv75_out", "conv76_weight", "conv76_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv76_out");

    builder.addTensor("conv77_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv77_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv77", "conv76_out", "conv77_weight", "conv77_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv77_out");

    builder.eltwise("add77", "add71_out", "conv77_out", ANEURALNETWORKS_FUSED_RELU,
                        "add77_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv78_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv78_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv78", "add77_out", "conv78_weight", "conv78_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv78_out");

    builder.addTensor("conv79_weight", {64, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv79_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv79", "conv78_out", "conv79_weight", "conv79_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv79_out");

    builder.addTensor("conv80_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv80_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv80", "conv79_out", "conv80_weight", "conv80_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv80_out");

    builder.addTensor("conv81_weight", {64, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv81_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv81", "conv80_out", "conv81_weight", "conv81_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv81_out");

    builder.addTensor("conv82_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv82_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv82", "conv81_out", "conv82_weight", "conv82_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv82_out");

    builder.addTensor("conv83_weight", {256, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv83_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv83", "conv82_out", "conv83_weight", "conv83_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv83_out");

    builder.addTensor("conv84_weight", {1, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv84_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv84", "add77_out", "conv84_weight", "conv84_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv84_out");

    builder.addTensor("conv85_weight", {256, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv85_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv85", "conv84_out", "conv85_weight", "conv85_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv85_out");

    builder.eltwise("add85", "conv83_out", "conv85_out", ANEURALNETWORKS_FUSED_RELU,
                        "add85_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv86_weight", {1, 3, 3, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv86_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv86", "add85_out", "conv86_weight", "conv86_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv86_out");

    builder.addTensor("conv87_weight", {64, 1, 1, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv87_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv87", "conv86_out", "conv87_weight", "conv87_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv87_out");

    builder.addTensor("conv88_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv88_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv88", "conv87_out", "conv88_weight", "conv88_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv88_out");

    builder.addTensor("conv89_weight", {64, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv89_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv89", "conv88_out", "conv89_weight", "conv89_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv89_out");

    builder.addTensor("conv90_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv90_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv90", "conv89_out", "conv90_weight", "conv90_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv90_out");

    builder.addTensor("conv91_weight", {256, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv91_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv91", "conv90_out", "conv91_weight", "conv91_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv91_out");

    builder.eltwise("add91", "add85_out", "conv91_out", ANEURALNETWORKS_FUSED_RELU,
                        "add91_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv92_weight", {1, 3, 3, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv92_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv92", "add91_out", "conv92_weight", "conv92_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv92_out");

    builder.addTensor("conv93_weight", {64, 1, 1, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv93_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv93", "conv92_out", "conv93_weight", "conv93_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv93_out");

    builder.addTensor("conv94_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv94_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv94", "conv93_out", "conv94_weight", "conv94_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv94_out");

    builder.addTensor("conv95_weight", {64, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv95_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv95", "conv94_out", "conv95_weight", "conv95_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv95_out");

    builder.addTensor("conv96_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv96_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv96", "conv95_out", "conv96_weight", "conv96_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv96_out");

    builder.addTensor("conv97_weight", {256, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv97_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv97", "conv96_out", "conv97_weight", "conv97_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv97_out");

    builder.eltwise("add97", "add91_out", "conv97_out", ANEURALNETWORKS_FUSED_RELU,
                        "add97_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    builder.addTensor("conv98_weight", {1, 3, 3, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv98_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv98", "add97_out", "conv98_weight", "conv98_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv98_out");

    builder.addTensor("conv99_weight", {64, 1, 1, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv99_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv99", "conv98_out", "conv99_weight", "conv99_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv99_out");

    builder.addTensor("conv100_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv100_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv100", "conv99_out", "conv100_weight", "conv100_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv100_out");

    builder.addTensor("conv101_weight", {64, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv101_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv101", "conv100_out", "conv101_weight", "conv101_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv101_out");

    builder.addTensor("conv102_weight", {1, 3, 3, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv102_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv102", "conv101_out", "conv102_weight", "conv102_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv102_out");

    builder.addTensor("conv103_weight", {256, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv103_bias", {256}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv103", "conv102_out", "conv103_weight", "conv103_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv103_out");

    builder.eltwise("add103", "add97_out", "conv103_out", ANEURALNETWORKS_FUSED_RELU,
                        "add103_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);



    builder.addTensor("conv104_weight", {128, 3, 3, 256}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv104_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv104", "add103_out", "conv104_weight", "conv104_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv104_out");

    builder.reduce("reduce104", "conv104_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, "reduce104_out");

    builder.addTensor("conv105_weight", {128, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv105_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv105", "reduce104_out", "conv105_weight", "conv105_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv105_out");







    builder.setInputTensors("data", indataptr, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.setOutputTensors("conv105_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);

    builder.compile(deviceIndex);
    builder.execute();

    std::vector<void *> results = builder.getOutput();

    for (int32_t idx = 0; idx < 60; ++idx)
    {
        uint8_t *result0 = reinterpret_cast<uint8_t *>(results[0]);
        SLOG_INFO << (int)(result0[idx]) << std::endl;
    }




    delete [] indataptr;
    delete [] weightptr;
    delete [] biasptr;


    return 0;
}