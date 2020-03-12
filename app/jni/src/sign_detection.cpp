#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "sign_detection.hpp"
#include <memory>

static auto LOG = spdlog::stdout_color_mt("SIGN");

SignDet::SignDet(const nlohmann::json config)
{
    SLOG_INFO << "Init sign detection" << std::endl;
    platform = std::make_unique<SNPEContext>(config["snpe"]);
    netW = config["model"]["net_width"].get<int>();
    netH = config["model"]["net_height"].get<int>();
    meanR = config["preprocessors"]["mean"]["R"].get<float>();
    meanG = config["preprocessors"]["mean"]["G"].get<float>();
    meanB = config["preprocessors"]["mean"]["B"].get<float>();
    stdR = config["preprocessors"]["std"]["R"].get<float>();
    stdG = config["preprocessors"]["std"]["G"].get<float>();
    stdB = config["preprocessors"]["std"]["B"].get<float>();
}

SignDet::~SignDet(void)
{

}

void SignDet::run(const uint8_t *src)
{
    preprocessing(src);
    inference();
    postprocessing();
}

void SignDet::saveresult(void)
{

}

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
    std::vector<float*> output_data = platform->forwardNN();
}

void SignDet::postprocessing(void)
{

}
