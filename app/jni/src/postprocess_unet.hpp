#pragma once

#include <json.hpp>
#include "postprocess.hpp"

class PostprocessUNET : Postprocess
{
public:
    PostprocessUNET(const nlohmann::json config);
    ~PostprocessUNET(void);
    void setInput(std::vector<float *> featuremaps);
    void run();
private:
    int netW;
    int netH;
    int netWCut;
    int netHCut;
    float scoreTh;
    float *scoreMap;
};