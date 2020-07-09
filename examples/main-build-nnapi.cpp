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
    float *indataptr = new float[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 1.0f);

    // kernels
    float dummyWeightBuf[16 * 5 * 5 * 3];
    float dummyBiasBuf[16];
    std::fill(dummyWeightBuf, dummyWeightBuf + (16 * 5 * 5 * 3), 2.0f);
    std::fill(dummyBiasBuf, dummyBiasBuf + 16, 1.0f);

    // start to build
    builder.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS});
    builder.addTensor("conv2d_1_weight", {16, 5, 5, 3}, dummyWeightBuf);
    builder.addTensor("conv2d_1_bias", {16}, dummyBiasBuf);

    builder.conv2d("conv2d_1", "data", "conv2d_1_weight", "conv2d_1_bias",
                    2, 2, 2, 2, 1, 1, ANEURALNETWORKS_FUSED_NONE, false, 1, 1,
                    "conv2d_1_out");

    // set input/output
    builder.setInputOps("data", indataptr);
    builder.setOutputOps("conv2d_1_out");

    // compile
    builder.compile(0);

    // execute
    builder.execute();

    // validate
    std::vector<float *> results = builder.getOutput();

    for (int32_t idx = 0; idx < 10; ++idx)
    {
        SLOG_INFO << results[0][idx] << std::endl;
    }


    delete [] indataptr;

    SLOG_INFO << "fine" << std::endl;
    return 0;
}