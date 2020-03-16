#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <assert.h>
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

    // todo: add c++ fs to traverse all jpg pics in assigned folder

    const std::string IMPATH = signconfig["evaluate"]["image_path"].get<std::string>();
    for (int i = 0; i < 10; ++i)
    {
        std::string picpath = cv::format(IMPATH.c_str(), i+1);
        std::string picsavepath = cv::format("/data/local/tmp/pics/%d_net_big_sign.jpg", i+1);

        cv::Mat im = cv::imread(picpath);

        assert(im.empty() != true);

        signdet.run(im.data, im.cols, im.rows, picsavepath);
    }



    return 0;
}