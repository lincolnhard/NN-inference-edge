#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"


static auto LOG = spdlog::stdout_color_mt("MAIN");


std::promise<void> isExit;


void segFunc(nlohmann::json config)
{
    const std::string TRT_ENGINE_PATH = config["engine"].get<std::string>();
    const int EVALUATE_TIMES = config["times"].get<int>();


    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);

    
    int loopCount = 0;
    double timesum = 0.0;
    auto waitExitCmd = isExit.get_future();
    while (waitExitCmd.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        nvmodel.runAsync();


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
        loopCount += 1;

        // {
        //     std::lock_guard<std::mutex> lock(gMutex);
        //     if (isTimeToStop == true)
        //     {
        //         break;
        //     }
        // }
    }

    SLOG_INFO << "loopCount: " << loopCount << std::endl; 
    SLOG_INFO << "Segmentation FPS: " << 1.0 / (timesum / loopCount / 1000.0) << std::endl;
}

void detFunc(nlohmann::json config)
{
    const std::string TRT_ENGINE_PATH = config["engine"].get<std::string>();
    const int EVALUATE_TIMES = config["times"].get<int>();


    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);


    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {

        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        nvmodel.runAsync();


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }

    // tell the other thread to terminate
    // {
    // std::lock_guard<std::mutex> lock(gMutex);
    // isTimeToStop = true;
    // }
    isExit.set_value();

    SLOG_INFO << "Detection FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;
}

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


    auto segmentationConfig = config["bisenet"];
    auto detectionConfig = config["mobilenetssd"];

    std::thread segmentationThread(segFunc, segmentationConfig);
    std::thread detectionThread(detFunc, detectionConfig);

    segmentationThread.join();
    detectionThread.join();

    return 0;
}