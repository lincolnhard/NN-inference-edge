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
    gallopwave::ModelBuilder builder;

    builder.getSdkVersion();
    builder.getDevices();

    return 0;
}