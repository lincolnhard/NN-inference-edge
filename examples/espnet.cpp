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
    // cv::RNG rng(12345);
    const int PLANESIZE = im.cols * im.rows;
    std::vector<const float*> segptrs;
    std::vector<cv::Vec3b> colors;
    for (int clsIdx = 0; clsIdx < NUM_CLASSES; ++clsIdx)
    {
        segptrs.push_back(segbuf + clsIdx * PLANESIZE);
        // colors.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }
    colors.push_back(cv::Vec3b(0, 0, 0));       // 0: background
    colors.push_back(cv::Vec3b(0, 255, 0));     // 1: solid line
    colors.push_back(cv::Vec3b(0, 0, 255));     // 2: dashed line
    colors.push_back(cv::Vec3b(255, 255, 0));     // 3: barrier
    colors.push_back(cv::Vec3b(255, 0, 0));     // 4: pole

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


void outputSegResult(std::string outpath, const float* segbuf, std::vector<int> tensorShapes, const int MAX_CLASSES)
{
    const int OUT_TENSOR_H = tensorShapes[0];
    const int OUT_TENSOR_W = tensorShapes[1];
    const int OUT_TENSOR_SIZE = OUT_TENSOR_H * OUT_TENSOR_W;


    std::vector<const float*> segptrs;
    for (int clsIdx = 0; clsIdx < MAX_CLASSES; ++clsIdx)
    {
        segptrs.push_back(segbuf + clsIdx * OUT_TENSOR_SIZE);
    }


    cv::Mat outIm = cv::Mat(OUT_TENSOR_H, OUT_TENSOR_W, CV_8UC1, cv::Scalar(0));

    for (int pixIdx = 0; pixIdx < OUT_TENSOR_SIZE; ++pixIdx)
    {
        int cls = std::distance(segptrs.begin(), std::max_element(segptrs.begin(), segptrs.end(), [](const float *x, const float *y){return (*x < *y);}));
        if (cls != 0)
        {
            outIm.data[pixIdx] = cls;
        }

        for (int clsIdx = 0; clsIdx < MAX_CLASSES; ++clsIdx)
        {
            segptrs[clsIdx] += 1;
        }
    }

    cv::imwrite(outpath, outIm);
}


void outputDetResult(const std::string& outpath, cv::Mat &im, std::vector<std::vector<KeyPoint>> &result, std::vector<std::string>& CLASS_NAME)
{
    std::ofstream fout(outpath, std::ios::trunc);

    const int numClass = result.size();
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        std::vector<KeyPoint> &shapes = result[clsIdx];
        const int numShape = shapes.size();
        for (int sIdx = 0; sIdx < numShape; ++sIdx)
        {
            float confidence = shapes[sIdx].scoreForSort;
            float left = shapes[sIdx].vertexTL.x * im.cols;
            float top = shapes[sIdx].vertexTL.y * im.rows;
            float right = shapes[sIdx].vertexBR.x * im.cols;
            float bottom = shapes[sIdx].vertexBR.y * im.rows;

            fout << CLASS_NAME[clsIdx] << ' ' << confidence << ' ' <<
            left << ' ' << top << ' ' << right << ' ' << bottom << std::endl;
        }
    }

    fout.close();
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
    std::vector<std::string> CLASS_NAME = config["model"]["class_name"].get<std::vector<std::string>>();
    std::vector<std::string> IN_TENSOR_NAMES = config["trt"]["input_layer_name"].get<std::vector<std::string> >();
    std::vector<std::string> OUT_TENSOR_NAMES = config["trt"]["output_layer_name"].get<std::vector<std::string> >();
    std::vector<std::vector<int>> OUT_TENSOR_SHAPES = config["trt"]["output_layer_shape"].get<std::vector<std::vector<int>>>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string FOLDERPATH = config["evaluate"]["folder_path"].get<std::string>();
    const std::string IMEXT = config["evaluate"]["extension"].get<std::string>();


    PostprocessFCOS postprocesser(config["model"]);
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
                intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
                intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
                intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
            }

            nvmodel.run();

            const float* scoresTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[0]));
            const float* vertexTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[1]));
            const float* centernessTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[2]));
            const float* segTensor = static_cast<const float*>(nvmodel.getHostBuffer(OUT_TENSOR_NAMES[3]));

            std::vector<const float *> featuremaps {scoresTensor, centernessTensor, vertexTensor};
            auto fcosResult = postprocesser.run(featuremaps);

            // plotSegResult(imnet, segTensor, NUM_SEG_CLASSES);
            // cv::imwrite(picpath.replace_extension().string() + "_result.png", imnet);

            // plotResult(imnet, fcosResult, segTensor, NUM_SEG_CLASSES);
            // cv::imwrite("result.jpg", imnet);

            // std::string outpath = picpath.replace_extension().string() + ".txt";
            // SLOG_INFO << outpath << std::endl;
            // outputDetResult(outpath, im, fcosResult, CLASS_NAME);

            std::string outpath = picpath.replace_extension().string() + ".png";
            SLOG_INFO << outpath << std::endl;
            outputSegResult(outpath, segTensor, OUT_TENSOR_SHAPES[0], NUM_SEG_CLASSES);
        }
    }


    return 0;
}