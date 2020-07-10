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
    const uint32_t NET_WIDTH = 300;
    const uint32_t NET_HEIGHT = 300;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // input
    uint8_t *indataptr = new uint8_t[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 1);

    // kernels
    uint8_t dummyWeightBuf[16 * 5 * 5 * 3];
    int32_t dummyBiasBuf[16];
    std::fill(dummyWeightBuf, dummyWeightBuf + (16 * 5 * 5 * 3), 2);
    std::fill(dummyBiasBuf, dummyBiasBuf + 16, 1);

    // start to build
    builder.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.addTensor("conv2d_1_weight", {16, 5, 5, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, dummyWeightBuf, 0.1, 1);
    builder.addTensor("conv2d_1_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, dummyBiasBuf, 0.1, 1);

    builder.conv2d("conv2d_1", "data", "conv2d_1_weight", "conv2d_1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    2, 2, 2, 2, 1, 1, ANEURALNETWORKS_FUSED_NONE, false, 1, 1,
                    "conv2d_1_out", 0.5, 100);

    // set input/output
    builder.setInputOps("data", indataptr, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.setOutputOps("conv2d_1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);

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

    SLOG_INFO << "fine" << std::endl;
    return 0;
}