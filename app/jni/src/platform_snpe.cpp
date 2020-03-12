#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "platform_snpe.hpp"
#include <string>
#include <vector>

static auto LOG = spdlog::stdout_color_mt("PLATFORM_SNPE");

SNPEContext::SNPEContext(const nlohmann::json config)
{
    std::string runtimeName = config["runtime"].get<std::string>();
    std::string dlcPath = config["dlc_path"].get<std::string>();
    outputLayerList = config["output_layer"].get<std::vector<std::string>>();
    zdl::DlSystem::StringList outputLayers;
    for (int i = 0; i < outputLayerList.size(); ++i)
    {
        outputLayers.append(outputLayerList[i].c_str());
    }


    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
    if (runtimeName == "gpu32")
    {
        runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
        SLOG_INFO << "running " << dlcPath << " on GPU" << std::endl;
    }
    else if (runtimeName == "dsp8")
    {
        runtime = zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
        SLOG_INFO << "running " << dlcPath << " on DSP" << std::endl;
    }
    else
    {
        runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
        SLOG_INFO << "running " << dlcPath << " on CPU" << std::endl;
    }


    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlcPath.c_str()));
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());


    snpengine = snpeBuilder.setOutputLayers(outputLayers)
                           .setRuntimeProcessor(runtime)
                           .setCPUFallbackMode(true)
                           .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE)
                           .build();
    zdl::DlSystem::TensorShape netTensorShape = snpengine->getInputDimensions();
    inTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(netTensorShape);
}

SNPEContext::~SNPEContext(void)
{
    snpengine.reset();
}

float *SNPEContext::getTensorPtr(void)
{
    return inTensor->begin().dataPointer();
}

std::vector<float*> SNPEContext::forwardNN(void)
{
    std::vector<float*> outptrs;
    zdl::DlSystem::TensorMap outTensors;
    snpengine->execute(inTensor.get(), outTensors);
    int numOutputLayers = outputLayerList.size();
    for (int i = 0; i < numOutputLayers; ++i)
    {
        outptrs.push_back(outTensors.getTensor(outputLayerList[i].c_str())->cbegin().dataPointer());
    }
    return outptrs;
}