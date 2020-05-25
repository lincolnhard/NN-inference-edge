#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

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

    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const float MEANR = config["model"]["mean"]["R"].get<float>();
    const float MEANG = config["model"]["mean"]["G"].get<float>();
    const float MEANB = config["model"]["mean"]["B"].get<float>();
    const float STDR = config["model"]["std"]["R"].get<float>();
    const float STDG = config["model"]["std"]["G"].get<float>();
    const float STDB = config["model"]["std"]["B"].get<float>();
    const std::string ONNXPATH = config["onnx"]["onnxpath"].get<std::string>();
    std::vector<std::string> OUT_TENSOR_NAMES = config["onnx"]["output_layer"].get<std::vector<std::string> >();
    const bool FP16MODE = config["caffe"]["fp16_mode"].get<bool>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();



    // gallopwave::NVModel nvmodel(ONNXPATH, FP16MODE);
    // nvmodel.outputEngine("data/espnetv2fusion.engine");

    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);

    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer("input"));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;

    cv::Mat im = cv::imread(IMPATH);
    cv::Mat imnet;
    cv::resize(im, imnet, cv::Size(NETW, NETH));
    for (int i = 0; i < NET_PLANESIZE; ++i)
    {
        intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
        intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
        intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
    }



    nvmodel.run();


    const float* scoresTensor = static_cast<const float*>(nvmodel.getHostBuffer("cls_score"));
    const float* centernessTensor = static_cast<const float*>(nvmodel.getHostBuffer("centerness"));
    const float* vertexTensor = static_cast<const float*>(nvmodel.getHostBuffer("bbox_pred"));
    const float* occlusionsTensor = static_cast<const float*>(nvmodel.getHostBuffer("occlusion"));
    const float* buoutTensor = static_cast<const float*>(nvmodel.getHostBuffer("bu_out"));

    SLOG_INFO << scoresTensor[0] << std::endl;
    SLOG_INFO << centernessTensor[0] << std::endl;
    SLOG_INFO << vertexTensor[0] << std::endl;
    SLOG_INFO << occlusionsTensor[0] << std::endl;
    SLOG_INFO << buoutTensor[0] << std::endl;



    return 0;
}