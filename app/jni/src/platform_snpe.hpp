#pragma once

#include <memory>
#include <json.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <string>
#include <memory>

class SNPEContext
{
public:
    SNPEContext(const nlohmann::json config);
    ~SNPEContext(void);
    float *getTensorPtr(void);
    std::vector<float*> forwardNN(void);
private:
    std::unique_ptr<zdl::SNPE::SNPE> snpengine;
    std::unique_ptr<zdl::DlSystem::ITensor> inTensor;
    std::vector<std::string> outputLayerList;
};
