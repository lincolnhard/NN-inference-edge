#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <experimental/filesystem>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"

static auto LOG = spdlog::stdout_color_mt("MAIN");


void plotResult(cv::Mat &im, const float* tensorBuf, std::vector<int> tensorShapes, const float TH)
{
    cv::RNG rng(12345);
    const int NUM_BOX = tensorShapes[0];
    const int NUM_BOX_ATTR = tensorShapes[1];

    const float* bufptr = tensorBuf;
    int numToDraw = 0;
    for (int box_i = 0; box_i < NUM_BOX; ++box_i)
    {
        auto left = bufptr[0] * (im.cols - 1);
        auto top = bufptr[1] * (im.rows - 1);
        auto right = bufptr[2] * (im.cols - 1);
        auto bottom = bufptr[3] * (im.rows - 1);
        auto confidence = bufptr[4];
        auto classType = bufptr[5];
        bufptr += NUM_BOX_ATTR;

        if ((classType == -1.0f) || (confidence < TH))
        {
            continue;
        }

        ++numToDraw;
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::rectangle(im, cv::Point(left, top), cv::Point(right, bottom), color, 1, 16);
    }
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
    const float DET_TH = config["model"]["class_score_threshold"].get<float>();
    const int NET_PLANESIZE = NETW * NETH;


    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);
    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer(IN_TENSOR_NAMES[0]));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;

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
            plotResult(imnet, scoresTensor, OUT_TENSOR_SHAPES[0], DET_TH);
            cv::imwrite(picpath.replace_extension().string() + "_result.jpg", imnet);
        }
    }


    return 0;
}