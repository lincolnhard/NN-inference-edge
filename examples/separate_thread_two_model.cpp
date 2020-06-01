#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"


static auto LOG = spdlog::stdout_color_mt("MAIN");

std::mutex gMutex;
bool isTimeToStop = false;



void segFunc(nlohmann::json config)
{
    const int NETW = config["net_width"].get<int>();
    const int NETH = config["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const float MEANR = config["mean"]["R"].get<float>();
    const float MEANG = config["mean"]["G"].get<float>();
    const float MEANB = config["mean"]["B"].get<float>();
    const float STDR = config["std"]["R"].get<float>();
    const float STDG = config["std"]["G"].get<float>();
    const float STDB = config["std"]["B"].get<float>();
    std::vector<std::string> IN_TENSOR_NAMES = config["input_layer"].get<std::vector<std::string> >();
    std::vector<std::string> OUT_TENSOR_NAMES = config["output_layer"].get<std::vector<std::string> >();
    const std::string TRT_ENGINE_PATH = config["engine"].get<std::string>();
    const std::string IMPATH = config["images_for_fps"].get<std::string>();
    const int EVALUATE_TIMES = config["times"].get<int>();



    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);
    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer(IN_TENSOR_NAMES[0]));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;


    int loopCount = 0;
    double timesum = 0.0;
    while (1)
    {
        cv::Mat imnet = cv::imread(cv::format(IMPATH.c_str(), (loopCount % 10) + 1));
        for (int i = 0; i < NET_PLANESIZE; ++i)
        {
            intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
            intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
            intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
        }


        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        nvmodel.runAsync();


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
        loopCount += 1;

        {
            std::lock_guard<std::mutex> lock(gMutex);
            if (isTimeToStop == true)
            {
                break;
            }
        }
    }


    SLOG_INFO << "Segmentation FPS: " << 1.0 / (timesum / loopCount / 1000.0) << std::endl;
}

void detFunc(nlohmann::json config)
{
    const int NETW = config["net_width"].get<int>();
    const int NETH = config["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const float MEANR = config["mean"]["R"].get<float>();
    const float MEANG = config["mean"]["G"].get<float>();
    const float MEANB = config["mean"]["B"].get<float>();
    const float STDR = config["std"]["R"].get<float>();
    const float STDG = config["std"]["G"].get<float>();
    const float STDB = config["std"]["B"].get<float>();
    std::vector<std::string> IN_TENSOR_NAMES = config["input_layer"].get<std::vector<std::string> >();
    std::vector<std::string> OUT_TENSOR_NAMES = config["output_layer"].get<std::vector<std::string> >();
    const std::string TRT_ENGINE_PATH = config["engine"].get<std::string>();
    const std::string IMPATH = config["images_for_fps"].get<std::string>();
    const int EVALUATE_TIMES = config["times"].get<int>();



    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);
    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer(IN_TENSOR_NAMES[0]));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;

    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        cv::Mat imnet = cv::imread(cv::format(IMPATH.c_str(), (t % 10) + 1));
        for (int i = 0; i < NET_PLANESIZE; ++i)
        {
            intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
            intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
            intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
        }


        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();


        nvmodel.runAsync();


        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }

    // tell the other thread to terminate
    {
    std::lock_guard<std::mutex> lock(gMutex);
    isTimeToStop = true;
    }

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


    auto segmentationConfig = config["rgp"];
    auto detectionConfig = config["mobilessd"];

    std::thread segmentationThread(segFunc, segmentationConfig);
    std::thread detectionThread(detFunc, detectionConfig);

    segmentationThread.join();
    detectionThread.join();

    return 0;
}