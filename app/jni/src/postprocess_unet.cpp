#include <assert.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "postprocess_unet.hpp"

static auto LOG = spdlog::stdout_color_mt("UNET");

PostprocessUNET::~PostprocessUNET()
{

}

PostprocessUNET::PostprocessUNET(const nlohmann::json config)
{
    netW = config["net_width"].get<int>();
    netH = config["net_height"].get<int>();
    netWCut = config["net_width_cut"].get<int>();
    netHCut = config["net_height_cut"].get<int>();
    scoreTh = config["score_threshold"].get<float>();
}

void PostprocessUNET::setInput(std::vector<float *> featuremaps)
{
    scoreMap = featuremaps[0];
}

void PostprocessUNET::run()
{
    const int PLANESIZE = netW * (netH - netWCut);
    for (int i = 0; i < PLANESIZE; ++i)
    {
        
    }
}