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
#include "postprocess_fcos.hpp"
#include "nv/run_model.hpp"

static auto LOG = spdlog::stdout_color_mt("MAIN");



void plotShapes(cv::Mat &im, std::vector<std::vector<KeyPoint>> &result, std::vector<int> num_vertex_byclass)
{
    const int numClass = result.size();
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        std::vector<KeyPoint> &shapes = result[clsIdx];
        const int numShape = shapes.size();
        const int numTotalVertex = num_vertex_byclass[clsIdx];
        for (int sIdx = 0; sIdx < numShape; ++sIdx)
        {
            cv::Point vertices[1][6];
            for (int i = 0; i < numTotalVertex; ++i)
            {
                vertices[0][i].x = shapes[sIdx].vertex[i].x * im.cols;
                vertices[0][i].y = shapes[sIdx].vertex[i].y * im.rows;
            }
            const cv::Point* vtsptr[1] = {vertices[0]};
            int npt[] = {numTotalVertex};
            cv::polylines(im, vtsptr, npt, 1, 1, cv::Scalar(0, 0, 255), 1, 16);
        }
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

    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const float MEANR = config["model"]["mean"]["R"].get<float>();
    const float MEANG = config["model"]["mean"]["G"].get<float>();
    const float MEANB = config["model"]["mean"]["B"].get<float>();
    const float STDR = config["model"]["std"]["R"].get<float>();
    const float STDG = config["model"]["std"]["G"].get<float>();
    const float STDB = config["model"]["std"]["B"].get<float>();
    const std::string CAFFETXT = config["caffe"]["prototxt"].get<std::string>();
    const std::string CAFFEBIN = config["caffe"]["caffemodel"].get<std::string>();
    std::vector<std::string> OUT_TENSOR_NAMES = config["caffe"]["output_layer"].get<std::vector<std::string> >();
    const bool FP16MODE = config["caffe"]["fp16_mode"].get<bool>();
    std::vector<int> NUM_VERTEX_BYCLASS = config["model"]["class_num_vertex"].get<std::vector<int> >();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();

    PostprocessFCOS postprocesser(config["model"]);


    // gallopwave::NVModel nvmodel(CAFFETXT, CAFFEBIN, OUT_TENSOR_NAMES, FP16MODE);
    // nvmodel.outputEngine("data/mnasneta1fcos.engine");

    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);

    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer("data"));
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


    const float* scoresTensor = static_cast<const float*>(nvmodel.getHostBuffer("scoremap_perm"));
    const float* centernessTensor = static_cast<const float*>(nvmodel.getHostBuffer("centernessmap_perm"));
    const float* vertexTensor = static_cast<const float*>(nvmodel.getHostBuffer("regressionmap_perm"));
    const float* occlusionsTensor = static_cast<const float*>(nvmodel.getHostBuffer("occlusionmap_perm"));


    std::vector<const float *> featuremaps {scoresTensor, centernessTensor, vertexTensor, occlusionsTensor};
    auto result = postprocesser.run(featuremaps);

    plotShapes(imnet, result, NUM_VERTEX_BYCLASS);
    cv::imwrite("result.jpg", imnet);


    return 0;
}