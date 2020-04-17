#pragma once

#include <json.hpp>

class PostprocessUNET
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