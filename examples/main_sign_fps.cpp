#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "sign_detection.hpp"

// using json = nlohmann::json;
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


    nlohmann::json signconfig = config["sign"];

    SignDet signdet(signconfig);

    const std::string IMPATH = signconfig["evaluate"]["image_path"].get<std::string>();
    const int NUM_RUN = signconfig["evaluate"]["times"].get<int>();
    double timesum = 0.0;
    for (int i = 0; i < NUM_RUN; ++i)
    {
        std::string picpath = cv::format(IMPATH.c_str(), (i % 10) + 1);
        cv::Mat im = cv::imread(picpath);

        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

        signdet.run(im.data);

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }

    SLOG_INFO << "sign detection fps: " << 1.0 / (timesum / NUM_RUN / 1000.0) << std::endl;



    return 0;
}