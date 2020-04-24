#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include <assert.h>
#include <dirent.h>
#include <json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"


static auto LOG = spdlog::stdout_color_mt("MAIN");


void plotResult(cv::Mat &im, const float *scoremap, const int NETH, const float SCORE_TH)
{
    const int PLANESIZE = im.cols * NETH;
    uint8_t *shiftsrc = im.data + (im.rows - NETH) * im.cols * 3;
    for (int i = 0; i < PLANESIZE; ++i)
    {
        if (scoremap[i] > SCORE_TH)
        {
            shiftsrc[i * 3 + 1] = 255;
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

    caffe::Net<float> net(config["caffe"]["prototxt"].get<std::string>(), caffe::TEST);
    net.CopyTrainedLayersFrom(config["caffe"]["caffemodel"].get<std::string>());


    const std::string folderpath = config["evaluate"]["folder_path"].get<std::string>();
    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NETPLANESIZE = NETW * NETH;
    const float SCORE_TH = config["model"]["score_threshold"].get<float>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    caffe::Blob<float>* intensor = net.input_blobs()[0];
    float *intensorPtrR = intensor->mutable_cpu_data();
    float *intensorPtrG = intensorPtrR + NETPLANESIZE;
    float *intensorPtrB = intensorPtrG + NETPLANESIZE;
    assert(NETW == intensor->width());
    assert(NETH == intensor->height());
    assert(3 == intensor->channels());
    std::vector<std::string> OUT_TENSOR_NAMES = config["caffe"]["output_layer"].get<std::vector<std::string> >();



    // TODO: need hardware resizing
    const float PRE_DIV = 1.0f / 255;
    cv::Mat im = cv::imread(IMPATH);
    cv::Mat imnet;
    cv::resize(im, imnet, cv::Size(NETW, 384)); // TODO: obtain from sign NETH
    const uint8_t *shiftsrc = imnet.data + (384 - NETH) * NETW * 3; // pass upper area, which is mostly sky
    for (int i = 0; i < NETPLANESIZE; ++i)
    {
        intensorPtrR[i] = shiftsrc[3 * i + 2] * PRE_DIV;
        intensorPtrG[i] = shiftsrc[3 * i + 1] * PRE_DIV;
        intensorPtrB[i] = shiftsrc[3 * i + 0] * PRE_DIV;
    }

    net.Forward();

    caffe::Blob<float>* scoresTensor = net.blob_by_name(OUT_TENSOR_NAMES[0]).get();
    const float* scoresptr = scoresTensor->cpu_data();

    plotResult(imnet, scoresptr, NETH, SCORE_TH);

    cv::imshow("demo", imnet);
    cv::waitKey(0);

    return 0;
}