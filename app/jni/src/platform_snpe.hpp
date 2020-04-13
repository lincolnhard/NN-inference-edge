#pragma once

#include <memory>
#include <json.hpp>
#include <string>
#include <memory>

class SNPEContext
{
public:
    SNPEContext(const nlohmann::json config, const int inputW, const int inputH);
    ~SNPEContext(void);
    float *getTensorPtr(void);
    std::vector<float*> forwardNN(void);
private:
    std::unique_ptr<zdl::SNPE::SNPE> snpengine;
    std::unique_ptr<zdl::DlSystem::ITensor> inTensor;
    std::vector<std::string> outputLayerList;
};
