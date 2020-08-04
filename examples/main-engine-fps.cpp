#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <cassert>
#include <utility>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"

static auto LOG = spdlog::stdout_color_mt("MAIN");



int main(int ac, char *av[])
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



    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();

    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);


    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

        nvmodel.run();
        // nvmodel.runAsync();

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }


    SLOG_INFO << "FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;

    return 0;
}