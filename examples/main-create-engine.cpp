#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

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


    const std::string ONNXPATH = config["onnx"]["onnxpath"].get<std::string>();
    std::vector<std::string> IN_TENSOR_NAMES = config["onnx"]["input_layer"].get<std::vector<std::string> >();
    std::vector<std::string> OUT_TENSOR_NAMES = config["onnx"]["output_layer"].get<std::vector<std::string> >();
    const bool FP16MODE = config["onnx"]["fp16_mode"].get<bool>();
    const std::string TRT_ENGINE_PATH = config["trt"]["engine"].get<std::string>();


    gallopwave::NVModel nvmodel(ONNXPATH, FP16MODE);
    nvmodel.outputEngine(TRT_ENGINE_PATH);

    return 0;
}