#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log_stream.hpp"
#include "nv/run_model.hpp"


static auto LOG = spdlog::stdout_color_mt("MAIN");

void TrtLogger::SetVerbose(bool const verbose) { verbose_ = verbose; }


int main(int ac, char *av[])
{
    if (ac != 2)
    {
        SLOG_ERROR << av[0] << " [config_file.json]" << std::endl;
        return 1;
    }

    TrtLogger trtlogger;
    cudaSetDevice(0); // TX2 has only one device
    initLibNvInferPlugins(&trtlogger, "");

    nlohmann::json config;
    std::ifstream fin;
    fin.open(av[1]);
    fin >> config;
    fin.close();



    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();
    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;

    std::ifstream engineFile(TRT_ENGINE_PATH, std::ios::binary);
    std::vector<char> engineFileStream(std::istreambuf_iterator<char>(engineFile), {});

    std::shared_ptr<nvinfer1::IRuntime> runtime =
        std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger), InferDeleter());
    std::shared_ptr<nvinfer1::ICudaEngine> engine =
        std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineFileStream.data(), engineFileStream.size(), nullptr), InferDeleter());
    std::shared_ptr<nvinfer1::IExecutionContext> context =
        std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext(), InferDeleter());

    LOG(INFO) << engine->getNbBindings() << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        LOG(INFO) << engine->getBindingName(i) << " " << engine->bindingIsInput(i) << std::endl;
        nvinfer1::Dims outDim = engine->getBindingDimensions(i);
        for (int j = 0; j < outDim.nbDims; ++j)
        {
            LOG(INFO) << outDim.d[j] << std::endl;
        }
    }

    BufferManager buffers(engine, 1);
    float* intensorPtrR = static_cast<float*>(buffers.getHostBuffer("transpose"));
    float* intensorPtrG = intensorPtrR + NET_PLANESIZE;
    float* intensorPtrB = intensorPtrG + NET_PLANESIZE;

    cv::Mat im = cv::imread(IMPATH);
    cv::Mat imnet;
    cv::resize(im, imnet, cv::Size(NETW, NETH));
    for (int i = 0; i < NET_PLANESIZE; ++i)
    {
        intensorPtrR[i] = imnet.data[3 * i + 2] / 255.0f;
        intensorPtrG[i] = imnet.data[3 * i + 1] / 255.0f;
        intensorPtrB[i] = imnet.data[3 * i + 0] / 255.0f;
    }

    gallopwave::NVModel nvmodel(TRT_ENGINE_PATH);

    // initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    // auto trtbuilder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    // auto trtnetwork = SampleUniquePtr<nvinfer1::INetworkDefinition>(trtbuilder->createNetwork());
    // auto trtconfig = SampleUniquePtr<nvinfer1::IBuilderConfig>(trtbuilder->createBuilderConfig());
    // auto trtparser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

    // // parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
    // // parser->registerOutput(mParams.outputTensorNames[0].c_str());
    // trtparser->parse(TRT_UFF_PATH.c_str(), *trtnetwork, nvinfer1::DataType::kFLOAT);
    // trtbuilder->setMaxBatchSize(1);
    // trtconfig->setMaxWorkspaceSize(1_GiB);

    // std::shared_ptr<nvinfer1::ICudaEngine> engine = 
    //     std::shared_ptr<nvinfer1::ICudaEngine>(trtbuilder->buildEngineWithConfig(*trtnetwork, *trtconfig), samplesCommon::InferDeleter());

    // // std::cout << engine->getBindingDimensions(0).d[0] << std::endl;
    // // std::cout << engine->getBindingDimensions(0).d[1] << std::endl;
    // // std::cout << engine->getBindingDimensions(0).d[2] << std::endl;

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }

    // nvuffparser::shutdownProtobufLibrary();

    return 0;
}