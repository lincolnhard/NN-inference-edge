#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <cassert>
#include <utility>

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "nv/run_model.hpp"

static auto LOG = spdlog::stdout_color_mt("MAIN");



int main(int ac, char *av[])
{
    if (ac != 2)
    {
        SLOG_ERROR << av[0] << " [config_file.json]" << std::endl;
        return 1;
    }

    nlohmann::json config;
    std::ifstream fin;
    fin.open(av[1]);
    fin >> config;
    fin.close();



    // const std::string UFFPATH = config["trt"]["uffpath"].get<std::string>();
    const std::string ONNXPATH = config["trt"]["onnxpath"].get<std::string>();



    std::vector<std::string> IN_TENSOR_NAMES = config["trt"]["input_layer_name"].get<std::vector<std::string>>();
    // std::vector<std::vector<int>> IN_TENSOR_SHAPES = config["trt"]["input_layer_shape"].get<std::vector<std::vector<int>>>();
    // assert(IN_TENSOR_NAMES.size() == IN_TENSOR_SHAPES.size());
    std::vector<std::string> OUT_TENSOR_NAMES = config["trt"]["output_layer_name"].get<std::vector<std::string>>();
    const bool FP16MODE = config["trt"]["fp16_mode"].get<bool>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();


    // std::vector<std::pair<std::string, nvinfer1::Dims>> uffInputs;
    // for (size_t i = 0; i < IN_TENSOR_NAMES.size(); ++i)
    // {
    //     nvinfer1::Dims dims;
    //     int numDimens = static_cast<int>(IN_TENSOR_SHAPES[i].size());
    //     dims.nbDims = numDimens;
    //     for (int j = 0; j < numDimens; ++j)
    //     {
    //         dims.d[j] = IN_TENSOR_SHAPES[i][j];
    //     }

    //     uffInputs.push_back(std::make_pair(IN_TENSOR_NAMES[i], dims));
    // }



    gallopwave::NVModel nvmodel(ONNXPATH, FP16MODE);
    // gallopwave::NVModel nvmodel(UFFPATH, uffInputs, OUT_TENSOR_NAMES, FP16MODE);

    nvmodel.outputEngine(TRT_ENGINE_PATH);

    return 0;
}