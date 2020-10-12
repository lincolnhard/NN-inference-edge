#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <experimental/filesystem>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"
#include "utils.hpp"

static auto LOG = spdlog::stdout_color_mt("MAIN");


void plotResult(cv::Mat &im, const float* tensorBuf, std::vector<int> tensorShapes, const int MAX_CLASSES)
{
    const int OUT_TENSOR_H = tensorShapes[0];
    const int OUT_TENSOR_W = tensorShapes[1];
    const int OUT_TENSOR_SIZE = OUT_TENSOR_H * OUT_TENSOR_W;

    auto tensorBufUint8 = std::shared_ptr<uint8_t[]>(new uint8_t[OUT_TENSOR_SIZE],
                                    [](uint8_t *p) { delete[] p; });
    std::transform(tensorBuf, tensorBuf + OUT_TENSOR_SIZE, tensorBufUint8.get(),
                                    [](float const f) { return static_cast<uint8_t>(f); });

    cv::Mat tensorIm = cv::Mat(OUT_TENSOR_H, OUT_TENSOR_W, CV_8UC1);
    std::copy(tensorBufUint8.get(), tensorBufUint8.get() + OUT_TENSOR_SIZE, tensorIm.data);

    // TODO: ugly here
    std::vector<cv::Scalar> labelColor;
    labelColor.push_back(cv::Scalar(0, 0, 0));      // 0: background
    labelColor.push_back(cv::Scalar(0, 0, 255));    // 1: dashed line
    labelColor.push_back(cv::Scalar(0, 255, 0));    // 2: solid line
    labelColor.push_back(cv::Scalar(255, 0, 0));    // 3: pole
    labelColor.push_back(cv::Scalar(255, 255, 0));  // 4: barrier

    for (int i = 1; i < MAX_CLASSES; ++i) // 0 for background, pass
    {
        cv::Mat currentClass = (tensorIm == i);
        if (cv::countNonZero(currentClass))
        {
            std::vector<std::vector<cv::Point>> contoursInteger;
            cv::findContours(currentClass, contoursInteger, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(im, contoursInteger, -1, labelColor[i], cv::FILLED);
        }
    }
}


void outputResult(std::string outpath, const float* tensorBuf, std::vector<int> tensorShapes, const int MAX_CLASSES)
{
    const int OUT_TENSOR_H = tensorShapes[0];
    const int OUT_TENSOR_W = tensorShapes[1];
    const int OUT_TENSOR_SIZE = OUT_TENSOR_H * OUT_TENSOR_W;

    auto tensorBufUint8 = std::shared_ptr<uint8_t[]>(new uint8_t[OUT_TENSOR_SIZE],
                                    [](uint8_t *p) { delete[] p; });
    std::transform(tensorBuf, tensorBuf + OUT_TENSOR_SIZE, tensorBufUint8.get(),
                                    [](float const f) { return static_cast<uint8_t>(f); });

    cv::Mat tensorIm = cv::Mat(OUT_TENSOR_H, OUT_TENSOR_W, CV_8UC1);
    std::copy(tensorBufUint8.get(), tensorBufUint8.get() + OUT_TENSOR_SIZE, tensorIm.data);

    cv::Mat outIm = cv::Mat(OUT_TENSOR_H, OUT_TENSOR_W, CV_8UC1, cv::Scalar(0));
    // TODO: ugly here
    std::vector<cv::Scalar> labels;
    labels.push_back(cv::Scalar(0)); // 0: nie background   ->      0: nit background
    labels.push_back(cv::Scalar(2)); // 1: nie dashed line  ->      2: nit dashed line
    labels.push_back(cv::Scalar(1)); // 2: nie solid line   ->      1: nit solid line
    labels.push_back(cv::Scalar(4)); // 3: nie pole         ->      4: nit pole
    labels.push_back(cv::Scalar(3)); // 4: nie barrier      ->      3: nit barrier

    for (int i = 1; i < MAX_CLASSES; ++i) // 0 for background, pass
    {
        cv::Mat currentClass = (tensorIm == i);
        if (cv::countNonZero(currentClass))
        {
            std::vector<std::vector<cv::Point>> contoursInteger;
            cv::findContours(currentClass, contoursInteger, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(outIm, contoursInteger, -1, labels[i], cv::FILLED);
        }
    }

    cv::imwrite(outpath, outIm);
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


    const std::string FOLDERPATH = config["evaluate"]["folder_path"].get<std::string>();
    const std::string IMEXT = config["evaluate"]["extension"].get<std::string>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();

    std::vector<std::string> IN_TENSOR_NAMES = config["trt"]["input_layer_name"].get<std::vector<std::string> >();
    std::vector<std::vector<int>> IN_TENSOR_SHAPES = config["trt"]["input_layer_shape"].get<std::vector<std::vector<int>>>();
    std::vector<std::string> OUT_TENSOR_NAMES = config["trt"]["output_layer_name"].get<std::vector<std::string> >();
    std::vector<std::vector<int>> OUT_TENSOR_SHAPES = config["trt"]["output_layer_shape"].get<std::vector<std::vector<int>>>();

    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const int NET_MAX_CLASSSES = config["model"]["max_classes"].get<int>();



    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);
    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer(IN_TENSOR_NAMES[0]));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;


    gallopwave::SysRes sysres;
    sysres.markCpuState();

    for (auto& picpathIt: std::experimental::filesystem::recursive_directory_iterator(FOLDERPATH))
    {
        std::experimental::filesystem::path picpath = picpathIt.path();
        if (picpath.extension().string() == IMEXT)
        {
            cv::Mat im = cv::imread(picpath.string());
            cv::Mat imnet;
            cv::resize(im, imnet, cv::Size(NETW, NETH));
            for (int i = 0; i < NET_PLANESIZE; ++i)
            {
                intensorPtrR[i] = imnet.data[3 * i + 2];
                intensorPtrG[i] = imnet.data[3 * i + 1];
                intensorPtrB[i] = imnet.data[3 * i + 0];
            }

            nvmodel.run();

            const float* scoresTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[0]));

            // plotResult(imnet, scoresTensor, OUT_TENSOR_SHAPES[0], NET_MAX_CLASSSES);
            // cv::imwrite(picpath.replace_extension().string() + "_result.jpg", imnet);

            // std::string outpath = picpath.replace_extension().string() + ".png";
            // SLOG_INFO << outpath << std::endl;
            // outputResult(outpath, scoresTensor, OUT_TENSOR_SHAPES[0], NET_MAX_CLASSSES);
        }
    }

    float cpusage = sysres.getCpuUsage();
    SLOG_INFO << "CPU usage = " << cpusage << std::endl;

    return 0;
}