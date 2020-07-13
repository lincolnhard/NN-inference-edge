#include <iostream>
#include "napi/builder.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"




static auto LOG = spdlog::stdout_color_mt("MAIN");

int main()
{
    gallopwave::ModelBuilder builder;

    builder.getSdkVersion();
    builder.getDevices();

    // NNAPI default data layout NHWC
    const uint32_t NET_WIDTH = 512;
    const uint32_t NET_HEIGHT = 512;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // input
    uint8_t *indataptr = new uint8_t[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 1);

    // kernels
    const uint32_t DUMMY_N = 100;
    const uint32_t DUMMY_C = 100;
    const uint32_t DUMMY_H = 100;
    const uint32_t DUMMY_W = 100;
    const uint32_t DUMMY_KERNEL_SIZE = DUMMY_N * DUMMY_H * DUMMY_W * DUMMY_C;
    uint8_t *weightptr = new uint8_t[DUMMY_KERNEL_SIZE];
    std::fill(weightptr, weightptr + DUMMY_KERNEL_SIZE, 2);
    int32_t *biasptr = new int32_t[DUMMY_N];
    std::fill(biasptr, biasptr + DUMMY_N, 1);


    // start to build
    builder.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.addTensor("conv1_weight", {32, 3, 3, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv1_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv1", "data", "conv1_weight", "conv1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, false, ANEURALNETWORKS_FUSED_RELU6, "conv1_out");

    builder.addTensor("conv2_weight", {32, 3, 3, 1}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv2_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv2", "conv1_out", "conv2_weight", "conv2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv2_out");

    builder.addTensor("conv3_weight", {16, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv3_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv3", "conv2_out", "conv3_weight", "conv3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv3_out");

    builder.addTensor("conv4_weight", {96, 1, 1, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv4_bias", {96}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv4", "conv3_out", "conv4_weight", "conv4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv4_out");

    builder.addTensor("conv5_weight", {96, 3, 3, 1}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv5_bias", {96}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv5", "conv4_out", "conv5_weight", "conv5_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_RELU6, "conv5_out");

    builder.addTensor("conv6_weight", {24, 1, 1, 96}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv6_bias", {24}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv6", "conv5_out", "conv6_weight", "conv6_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv6_out");

    builder.addTensor("conv7_weight", {144, 1, 1, 24}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv7_bias", {144}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv7", "conv6_out", "conv7_weight", "conv7_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv7_out");

    builder.addTensor("conv8_weight", {144, 3, 3, 1}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv8_bias", {144}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv8", "conv7_out", "conv8_weight", "conv8_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_RELU6, "conv8_out");

    builder.addTensor("conv9_weight", {24, 1, 1, 144}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv9_bias", {24}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv9", "conv8_out", "conv9_weight", "conv9_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv9_out");







    // set input/output
    builder.setInputOps("data", indataptr, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.setOutputOps("conv9_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);

    // compile
    builder.compile(1);

    // execute
    builder.execute();

    // validate
    std::vector<void *> results = builder.getOutput();

    for (int32_t idx = 0; idx < 10; ++idx)
    {
        uint8_t *result0 = reinterpret_cast<uint8_t *>(results[0]);
        SLOG_INFO << (int)(result0[idx]) << std::endl;
    }


    delete [] indataptr;
    delete [] weightptr;
    delete [] biasptr;

    SLOG_INFO << "fine" << std::endl;
    return 0;
}