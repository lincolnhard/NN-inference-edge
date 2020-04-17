#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <caffe/caffe.hpp>
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <assert.h>
#include "log_stream.hpp"
#include "postprocess_fcos.hpp"

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
            cv::polylines(im, vtsptr, npt, 1, 1, cv::Scalar(0, 0, 255), 2, 8);
        }
    }
}

int main (int ac, char *av[])
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

    PostprocessFCOS postprocesser(config["model"]);

    caffe::Net<float> net(config["caffe"]["prototxt"].get<std::string>(), caffe::TEST);
    net.CopyTrainedLayersFrom(config["caffe"]["caffemodel"].get<std::string>());


    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NETPLANESIZE = NETW * NETH;
    assert(NETW == intensor->width());
    assert(NETH == intensor->height());
    assert(3 == intensor->channels());
    caffe::Blob<float>* intensor = net.input_blobs()[0];
    float *intensorPtrR = intensor->mutable_cpu_data();
    float *intensorPtrG = intensorPtrR + NETPLANESIZE;
    float *intensorPtrB = intensorPtrG + NETPLANESIZE;
    const float MEANR = config["model"]["mean"]["R"].get<float>();
    const float MEANG = config["model"]["mean"]["G"].get<float>();
    const float MEANB = config["model"]["mean"]["B"].get<float>();
    const float STDR = config["model"]["std"]["R"].get<float>();
    const float STDG = config["model"]["std"]["G"].get<float>();
    const float STDB = config["model"]["std"]["B"].get<float>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    std::vector<int> NUM_VERTEX_BYCLASS = config["model"]["class_num_vertex"].get<std::vector<int> >();


    // TODO: need optimization
    cv::Mat im = cv::imread(IMPATH);
    cv::Mat imnet;
    cv::resize(im, imnet, cv::Size(NETW, NETH));
    for (int i = 0; i < NETPLANESIZE; ++i)
    {
        intensorPtrR[i] = ((imnet.data[3 * i + 2] / 255.0f) - MEANR) / STDR;
        intensorPtrG[i] = ((imnet.data[3 * i + 1] / 255.0f) - MEANG) / STDG;
        intensorPtrB[i] = ((imnet.data[3 * i + 0] / 255.0f) - MEANB) / STDB;
    }

    net.Forward();


    caffe::Blob<float>* scoresTensor = net.blob_by_name("scoremap_perm").get();
    caffe::Blob<float>* centernessTensor = net.blob_by_name("centernessmap_perm").get();
    caffe::Blob<float>* vertexTensor = net.blob_by_name("regressionmap_perm").get();
    caffe::Blob<float>* occlusionsTensor = net.blob_by_name("occlusionmap_perm").get();
    const float* scoresptr = scoresTensor->cpu_data();
    const float* centernessptr = centernessTensor->cpu_data();
    const float* vertexptr = vertexTensor->cpu_data();
    const float* occlusionsptr = occlusionsTensor->cpu_data();
    std::vector<const float *> featuremaps {scoresptr, centernessptr, vertexptr, occlusionsptr};

    auto result = postprocesser.run(featuremaps);

    plotShapes(im, result, NUM_VERTEX_BYCLASS);

    cv::imshow("demo", im);
    cv::waitKey(0);

    return 0;
}