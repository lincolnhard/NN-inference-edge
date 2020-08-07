#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"
#include "postprocess_fcos.hpp"

static auto LOG = spdlog::stdout_color_mt("MAIN");


void plotResult(cv::Mat &im, std::vector<std::vector<KeyPoint>> &result, const float* segbuf, int numSegClass)
{
    cv::RNG rng(12345);
    const int PLANESIZE = im.cols * im.rows;
    std::vector<const float*> segptrs;
    std::vector<cv::Vec3b> colors;
    for (int clsIdx = 0; clsIdx < numSegClass; ++clsIdx)
    {
        segptrs.push_back(segbuf + clsIdx * PLANESIZE);
        colors.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }
    for (int pixIdx = 0; pixIdx < PLANESIZE; ++pixIdx)
    {
        int cls = std::distance(segptrs.begin(), std::max_element(segptrs.begin(), segptrs.end(), [](const float *x, const float *y){return (*x < *y);}));
        if (cls != 0)
        {
            im.at<cv::Vec3b>(pixIdx) = colors[cls];
        }
        for (int clsIdx = 0; clsIdx < numSegClass; ++clsIdx)
        {
            segptrs[clsIdx] += 1;
        }
    }


    const int numBboxClass = result.size();
    for (int clsIdx = 0; clsIdx < numBboxClass; ++clsIdx)
    {
        std::vector<KeyPoint> &shapes = result[clsIdx];
        const int numShape = shapes.size();
        for (int sIdx = 0; sIdx < numShape; ++sIdx)
        {
            cv::Point vertices[2];
            vertices[0].x = shapes[sIdx].vertexTL.x * im.cols;
            vertices[0].y = shapes[sIdx].vertexTL.y * im.rows;
            vertices[1].x = shapes[sIdx].vertexBR.x * im.cols;
            vertices[1].y = shapes[sIdx].vertexBR.y * im.rows;

            cv::rectangle(im, vertices[0], vertices[1], cv::Scalar(0, 0, 255), 1, 16);
        }
    }
}


void plotSegResult(cv::Mat &im, const float* segbuf, int NUM_CLASSES)
{
    cv::RNG rng(12345);
    const int PLANESIZE = im.cols * im.rows;
    std::vector<const float*> segptrs;
    std::vector<cv::Vec3b> colors;
    for (int clsIdx = 0; clsIdx < NUM_CLASSES; ++clsIdx)
    {
        segptrs.push_back(segbuf + clsIdx * PLANESIZE);
        colors.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }

    for (int pixIdx = 0; pixIdx < PLANESIZE; ++pixIdx)
    {
        int cls = std::distance(segptrs.begin(), std::max_element(segptrs.begin(), segptrs.end(), [](const float *x, const float *y){return (*x < *y);}));
        if (cls != 0)
        {
            im.at<cv::Vec3b>(pixIdx) = colors[cls];
        }
        for (int clsIdx = 0; clsIdx < NUM_CLASSES; ++clsIdx)
        {
            segptrs[clsIdx] += 1;
        }
    }
}


void plotBboxResult(cv::Mat &im, std::vector<std::vector<KeyPoint>> &result)
{
    const int numClass = result.size();
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        std::vector<KeyPoint> &shapes = result[clsIdx];
        const int numShape = shapes.size();
        for (int sIdx = 0; sIdx < numShape; ++sIdx)
        {
            cv::Point vertices[2];
            vertices[0].x = shapes[sIdx].vertexTL.x * im.cols;
            vertices[0].y = shapes[sIdx].vertexTL.y * im.rows;
            vertices[1].x = shapes[sIdx].vertexBR.x * im.cols;
            vertices[1].y = shapes[sIdx].vertexBR.y * im.rows;

            cv::rectangle(im, vertices[0], vertices[1], cv::Scalar(0, 0, 255), 1, 16);
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
    const int NUM_SEG_CLASSES = config["model"]["num_class_seg"].get<int>();
    std::vector<std::string> IN_TENSOR_NAMES = config["trt"]["input_layer_name"].get<std::vector<std::string> >();
    std::vector<std::string> OUT_TENSOR_NAMES = config["trt"]["output_layer_name"].get<std::vector<std::string> >();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();


    PostprocessFCOS postprocesser(config["model"]);
    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);


    float* intensorPtrR = static_cast<float*>(nvmodel.getHostBuffer(IN_TENSOR_NAMES[0]));
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



    const float* scoresTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[0]));
    const float* vertexTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[1]));
    const float* centernessTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[2]));
    const float* segTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[3]));

    // SLOG_INFO << scoresTensor[0] << ',' << scoresTensor[1] << ',' << scoresTensor[2] << ',' << scoresTensor[3] << ',' << scoresTensor[4] << std::endl;
    // SLOG_INFO << vertexTensor[0] << ',' << vertexTensor[1] << ',' << vertexTensor[2] << ',' << vertexTensor[3] << ',' << vertexTensor[4] << std::endl;
    // SLOG_INFO << centernessTensor[0] << ',' << centernessTensor[1] << ',' << centernessTensor[2] << ',' << centernessTensor[3] << ',' << centernessTensor[4] << std::endl;
    // SLOG_INFO << segTensor[0] << ',' << segTensor[1] << ',' << segTensor[2] << ',' << segTensor[3] << ',' << segTensor[4] << std::endl;

    std::vector<const float *> featuremaps {scoresTensor, centernessTensor, vertexTensor};
    auto fcosResult = postprocesser.run(featuremaps);




    // cv::Mat imnet2 = imnet.clone();
    // plotBboxResult(imnet, fcosResult);
    // cv::imwrite("bbox.jpg", imnet);
    // plotSegResult(imnet2, segTensor, NUM_SEG_CLASSES);
    // cv::imwrite("seg.jpg", imnet2);

    plotResult(imnet, fcosResult, segTensor, NUM_SEG_CLASSES);
    cv::imwrite("result.jpg", imnet);

    return 0;
}