#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "lane_detection.hpp"
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static auto LOG = spdlog::stdout_color_mt("LANE");

LaneDet::LaneDet(const nlohmann::json config)
{
    SLOG_INFO << "Init lane detection start" << std::endl;
    netW = config["model"]["net_width"].get<int>();
    netH = config["model"]["net_height"].get<int>();
    netWCut = config["model"]["net_width_cut"].get<int>();
    netHCut = config["model"]["net_height_cut"].get<int>();
    scoreTh = config["model"]["score_threshold"].get<float>();

    platform = std::make_unique<SNPEContext>(config["snpe"], netW - netWCut, netH - netHCut);
    postprocessor = std::make_unique<PostprocessUNET>(config["model"]);

    SLOG_INFO << "Init lane detection done" << std::endl;
}

LaneDet::~LaneDet(void)
{

}

float* LaneDet::run(uint8_t *src)
{
    preprocessing(src);
    inference();
    postprocessing();

    // saveresult(src, srcw, srch, dstpath);
    return scoreMap;
}

void LaneDet::preprocessing(const uint8_t *src)
{
    // todo: not consider netWCut here
    const float PRE_DIV = 1.0f / 255;
    const int NUM_PIXELS = (netW - netWCut) * (netH - netHCut) * 3;
    const uint8_t *shiftsrc = src + netHCut * netW * 3; // pass upper area, which is mostly sky
    float *intensorptr = platform->getTensorPtr();
    // SNPE take RGB
    for (int i = 0; i < NUM_PIXELS; ++i)
    {
        intensorptr[i] = (shiftsrc[i]) * PRE_DIV;
    }
}

void LaneDet::inference(void)
{
    outputMaps = platform->forwardNN();
    // postprocessor->setInput(outputMaps);
}

void LaneDet::postprocessing(void)
{
    scoreMap = outputMaps[0];
}

// void LaneDet::saveresult(uint8_t *src, int srcw, int srch, std::string dstpath)
// {
//     float *scoreMap = outputMaps[0];
//     const int PLANESIZE = netW * (netH - netHCut);
//     uint8_t *shiftsrc = src + netHCut * netW * 3;

//     for (int i = 0; i < PLANESIZE; ++i)
//     {
//         if (scoreMap[i] > scoreTh)
//         {
//             shiftsrc[i * 3 + 1] = 255;
//         }
//     }

//     cv::Mat dst(srch, srcw, CV_8UC3, src);
//     cv::imwrite(dstpath, dst);
// }