#pragma once
#include "platform_snpe.hpp"
#include "postprocess_fcos.hpp"
#include <json.hpp>
#include <cstdint>
#include <string>


class SignDet
{
public:
    SignDet(const nlohmann::json config);
    ~SignDet(void);
    std::vector<std::vector<ScoreVertices>> run(uint8_t *src);
    // void saveresult(uint8_t *src, const int srcw, const int srch, std::string dstpath);
    std::vector<std::vector<ScoreVertices>> predResult;
private:
    void preprocessing(const uint8_t *src);
    void inference(void);
    void postprocessing(void);
    void reprojectBasedOnTemplate(void);

    std::unique_ptr<SNPEContext> platform;
    std::unique_ptr<PostprocessFCOS> postprocessor;
    int netW;
    int netH;
    int numClass;
    float meanR;
    float meanG;
    float meanB;
    float stdR;
    float stdG;
    float stdB;
};