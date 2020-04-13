#pragma once
#include <cstdint>
#include <string>
#include "platform_snpe.hpp"
#include "postprocess_unet.hpp"
#include <json.hpp>

class LaneDet
{
public:
    LaneDet(const nlohmann::json config);
    ~LaneDet(void);
    float *run(uint8_t *src);
    // void saveresult(uint8_t *src, const int srcw, const int srch, std::string dstpath);
private:
    void preprocessing(const uint8_t *src);
    void inference(void);
    void postprocessing(void);

    std::unique_ptr<SNPEContext> platform;
    std::unique_ptr<PostprocessUNET> postprocessor;
    std::vector<float*> outputMaps;
    float *scoreMap;
    int netW;
    int netH;
    int netWCut;
    int netHCut;
    float scoreTh;
};