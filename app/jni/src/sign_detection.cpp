#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "sign_detection.hpp"
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static auto LOG = spdlog::stdout_color_mt("SIGN");

SignDet::SignDet(const nlohmann::json config)
{
    SLOG_INFO << "Init sign detection start" << std::endl;
    netW = config["model"]["net_width"].get<int>();
    netH = config["model"]["net_height"].get<int>();
    numClass = config["model"]["num_class"].get<int>();
    meanR = config["preprocessors"]["mean"]["R"].get<float>();
    meanG = config["preprocessors"]["mean"]["G"].get<float>();
    meanB = config["preprocessors"]["mean"]["B"].get<float>();
    stdR = config["preprocessors"]["std"]["R"].get<float>();
    stdG = config["preprocessors"]["std"]["G"].get<float>();
    stdB = config["preprocessors"]["std"]["B"].get<float>();

    platform = std::make_unique<SNPEContext>(config["snpe"], netW, netH);
    postprocessor = std::make_unique<PostprocessFCOS>(config["model"]);
    predResult.resize(numClass);
}

SignDet::~SignDet(void)
{

}

std::vector<std::vector<ScoreVertices>> SignDet::run(uint8_t *src)
{
    preprocessing(src);
    inference();
    postprocessing();

    reprojectBasedOnTemplate();
    // saveresult(src, srcw, srch, dstpath);

    return predResult;
}

// void SignDet::saveresult(uint8_t *src, int srcw, int srch, std::string dstpath)
// {
//     cv::Mat dst(srch, srcw, CV_8UC3, src);
//     for (int clsIdx = 0; clsIdx < numClass - 1; ++clsIdx)
//     {
//         std::vector<ScoreVertices> &signobjvector = predResult[clsIdx];
//         const int numSign = signobjvector.size();
//         for (int signIdx = 0; signIdx < numSign; ++signIdx)
//         {
//             cv::Point vertices[1][4];
//             vertices[0][0] = cv::Point(signobjvector[signIdx].x0 * srcw, signobjvector[signIdx].y0 * srch);
//             vertices[0][1] = cv::Point(signobjvector[signIdx].x1 * srcw, signobjvector[signIdx].y1 * srch);
//             vertices[0][2] = cv::Point(signobjvector[signIdx].x2 * srcw, signobjvector[signIdx].y2 * srch);
//             vertices[0][3] = cv::Point(signobjvector[signIdx].x3 * srcw, signobjvector[signIdx].y3 * srch);
//             const cv::Point* vtsptr[1] = {vertices[0]};
//             int npt[] = {4};
//             cv::polylines(dst, vtsptr, npt, 1, 1, cv::Scalar(255, 0, 0), 2, 16);
//         }
//     }
//     cv::imwrite(dstpath, dst);
// }

void SignDet::preprocessing(const uint8_t *src)
{
    const float PRE_DIV = 1.0f / 255;
    const int PLANESIZE = netW * netH;
    float *intensorptr = platform->getTensorPtr();
    const uint8_t *srcptr = src;
    // SNPE take RGB
    for (int i = 0; i < PLANESIZE; ++i)
    {
        intensorptr[0] = (srcptr[2] * PRE_DIV - meanR) / stdR;
        intensorptr[1] = (srcptr[1] * PRE_DIV - meanG) / stdG;
        intensorptr[2] = (srcptr[0] * PRE_DIV - meanB) / stdB;
        srcptr += 3;
        intensorptr += 3;
    }
}

void SignDet::inference(void)
{
    std::vector<float*> outputMaps = platform->forwardNN();
    postprocessor->setInput(outputMaps);
}

void SignDet::postprocessing(void)
{
    postprocessor->run(predResult);
}

// maybe opencv should not appear here
void SignDet::reprojectBasedOnTemplate(void)
{
    // reprojection template
    float traffic4[] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float traffic4y[] = {0.5f, 0.0f, 1.0f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.5f, 1.0f};
    float traffic3[] = {0.5f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float traffic3rev[] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.5f, 1.0f, 1.0f};

    cv::Mat signtemplate[4];
    signtemplate[0] = cv::Mat(4, 3, CV_32FC1, traffic4);
    signtemplate[1] = cv::Mat(4, 3, CV_32FC1, traffic4y);
    signtemplate[2] = cv::Mat(3, 3, CV_32FC1, traffic3);
    signtemplate[3] = cv::Mat(3, 3, CV_32FC1, traffic3rev);

    // no need to reproject traffic4, start from clsIdx 1
    for (int clsIdx = 1; clsIdx < numClass; ++clsIdx)
    {
        const int numSign = predResult[clsIdx].size();
        std::vector<ScoreVertices> &signobjvector = predResult[clsIdx];

        for (int signIdx = 0; signIdx < numSign; ++signIdx)
        {
            cv::Mat srcA(4, 3, CV_32FC1);
            srcA.at<float>(0, 0) = signobjvector[signIdx].x0;
            srcA.at<float>(0, 1) = signobjvector[signIdx].y0;
            srcA.at<float>(0, 2) = 1.0f;
            srcA.at<float>(1, 0) = signobjvector[signIdx].x1;
            srcA.at<float>(1, 1) = signobjvector[signIdx].y1;
            srcA.at<float>(1, 2) = 1.0f;
            srcA.at<float>(2, 0) = signobjvector[signIdx].x2;
            srcA.at<float>(2, 1) = signobjvector[signIdx].y2;
            srcA.at<float>(2, 2) = 1.0f;
            srcA.at<float>(3, 0) = signobjvector[signIdx].x3;
            srcA.at<float>(3, 1) = signobjvector[signIdx].y3;
            srcA.at<float>(3, 2) = 1.0f;
            cv::Mat H;
            cv::solve(signtemplate[0], srcA, H, cv::DECOMP_QR); // [0] is fixed
            cv::Mat augProjectedSrcA = signtemplate[clsIdx] * H;
            float *projectedPolygon = (float *)augProjectedSrcA.data;
            signobjvector[signIdx].x0 = projectedPolygon[0];
            signobjvector[signIdx].y0 = projectedPolygon[1];
            signobjvector[signIdx].x1 = projectedPolygon[3];
            signobjvector[signIdx].y1 = projectedPolygon[4];
            signobjvector[signIdx].x2 = projectedPolygon[6];
            signobjvector[signIdx].y2 = projectedPolygon[7];
            if (clsIdx == 1)
            {
                signobjvector[signIdx].x3 = projectedPolygon[9]; // 4-points sign only
                signobjvector[signIdx].y3 = projectedPolygon[10]; // 4-points sign only
            }
        }
    }
}