#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <chrono>
#include <thread>

#include <rknn_api.h>
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "log_stream.hpp"



static auto LOG = spdlog::stdout_color_mt("MAIN");


#define CHECK(status)                                                                                                   \
    do                                                                                                                  \
    {                                                                                                                   \
        auto ret = (status);                                                                                            \
        if (ret < 0)                                                                                                    \
        {                                                                                                               \
            SLOG_ERROR << "RKNN failure: " << ret << ", line: " << __LINE__ << std::endl;                               \
            exit(1);                                                                                                    \
        }                                                                                                               \
    } while (0)


void printRknnEstimatedFPS(rknn_context ctx) // for reference only, not include CPU <-> NPU memcpy
{
    rknn_perf_run perf;
    CHECK( rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(rknn_perf_run)) );
    SLOG_INFO << "RKNN estimated FPS: " << 1000000.0 / perf.run_duration << std::endl;
}


void saveRknnInferenceDetail(rknn_context ctx) // should enable RKNN_FLAG_COLLECT_PERF_MASK first when creating context
{
    rknn_perf_detail detail;
    CHECK( rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(rknn_perf_detail)) );
    std::ofstream fout;
    fout.open("perf.txt", std::ios::out | std::ios::trunc);
    for (uint64_t i = 0; i < detail.data_len; ++i)
    {
        fout << detail.perf_data[i];
    }
    fout.close();
}


void getNumInOut(uint32_t& numInputs, uint32_t& numOutputs, rknn_context ctx)
{
    rknn_input_output_num rion;
    CHECK( rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &rion, sizeof(rknn_input_output_num)) );
    numInputs = rion.n_input;
    numOutputs = rion.n_output;
    SLOG_INFO << "Number of inputs: " << numInputs << std::endl;
    SLOG_INFO << "Number of outputs: " << numOutputs << std::endl;
}


void getInElems(uint32_t numTensors, uint32_t *elems, rknn_context ctx)
{
    rknn_tensor_attr *attrs = new rknn_tensor_attr [numTensors];
    for (uint32_t i = 0; i < numTensors; ++i)
    {
        attrs[i].index = i;
        CHECK( rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(attrs[i]), sizeof(rknn_tensor_attr)) );
    }

    for (uint32_t i = 0; i < numTensors; ++i)
    {
        elems[i] = attrs[i].n_elems;
        SLOG_INFO << "In " << i << " number of elements: " << attrs[i].n_elems << std::endl;
        SLOG_INFO << "In " << i << " size: " << attrs[i].size << std::endl;
        SLOG_INFO << "In " << i << " format: " << attrs[i].fmt << std::endl;
        SLOG_INFO << "In " << i << " type: " << attrs[i].type << std::endl;
        SLOG_INFO << "In " << i << " quantize type: " << attrs[i].qnt_type << std::endl;
    }
    delete [] attrs;
}


void getOutElems(uint32_t numTensors, uint32_t *elems, rknn_context ctx)
{
    rknn_tensor_attr *attrs = new rknn_tensor_attr [numTensors];
    for (uint32_t i = 0; i < numTensors; ++i)
    {
        attrs[i].index = i;
        CHECK( rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(attrs[i]), sizeof(rknn_tensor_attr)) );
    }

    for (uint32_t i = 0; i < numTensors; ++i)
    {
        elems[i] = attrs[i].n_elems;
        SLOG_INFO << "Out " << i << " number of elements: " << attrs[i].n_elems << std::endl;
        SLOG_INFO << "Out " << i << " size: " << attrs[i].size << std::endl;
        SLOG_INFO << "Out " << i << " format: " << attrs[i].fmt << std::endl;
        SLOG_INFO << "Out " << i << " type: " << attrs[i].type << std::endl;
        SLOG_INFO << "Out " << i << " quantize type: " << attrs[i].qnt_type << std::endl;
    }
    delete [] attrs;
}



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

    const int NETW = config["model"]["net_width"].get<int>();
    const int NETH = config["model"]["net_height"].get<int>();
    const int NETC = config["model"]["net_channels"].get<int>();
    const int NET_PLANESIZE = NETW * NETH;
    const std::string MODELPATH = config["rknn"]["modelpath"].get<std::string>();
    const std::string IMPATH = config["evaluate"]["image_path"].get<std::string>();
    const int EVALUATE_TIMES = config["evaluate"]["times"].get<int>();


    fin.open(MODELPATH, std::ios::in | std::ios::binary | std::ios::ate);
    const uint32_t MODELSIZE = fin.tellg();
    fin.seekg(0, std::ios::beg);
    char *modelbuf = new char [MODELSIZE];
    fin.read(modelbuf, MODELSIZE);
    fin.close();
    SLOG_INFO << "Loading RKNN model complete" << std::endl;



    rknn_context ctx = 0;
    CHECK( rknn_init(&ctx, modelbuf, MODELSIZE, RKNN_FLAG_PRIOR_HIGH) );



    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;
    getNumInOut(numInputs, numOutputs, ctx);
    uint32_t *numInputElems = new uint32_t [numInputs];
    uint32_t *numOutputElems = new uint32_t [numOutputs];
    getInElems(numInputs, numInputElems, ctx);
    getOutElems(numOutputs, numOutputElems, ctx);
    std::vector <uint8_t *> inbufs;
    std::vector <float *> outbufs;
    for (uint32_t i = 0; i < numInputs; ++i)
    {
        inbufs.push_back(new uint8_t [numInputElems[i]]);
    }
    for (uint32_t i = 0; i < numInputs; ++i)
    {
        outbufs.push_back(new float [numOutputElems[i]]);
    }


    rknn_input *rknnInputs = new rknn_input [numInputs];

    rknn_output *rknnOutputs = new rknn_output [numOutputs];

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        rknnInputs[i].index = i;
        rknnInputs[i].buf = inbufs[i];
        rknnInputs[i].size = numInputElems[i] * sizeof(uint8_t);
        rknnInputs[i].pass_through = true;
        // rknnInputs[i].type = RKNN_TENSOR_UINT8;
        // rknnInputs[i].fmt = RKNN_TENSOR_NCHW;
    }

    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        rknnOutputs[i].want_float = true;
        rknnOutputs[i].is_prealloc = false;
    }


    double timesum = 0.0;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();



        CHECK( rknn_inputs_set(ctx, numInputs, rknnInputs) );
        CHECK( rknn_run(ctx, nullptr) );
        CHECK( rknn_outputs_get(ctx, numOutputs, rknnOutputs, nullptr) );
        rknn_outputs_release(ctx, numOutputs, rknnOutputs);


        // float *cls_logits = reinterpret_cast<float *>(rknnOutputs[0].buf);
        // float *bbox_pred = reinterpret_cast<float *>(rknnOutputs[1].buf);

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }



    SLOG_INFO << "FPS: " << 1.0 / (timesum / EVALUATE_TIMES / 1000.0) << std::endl;

    printRknnEstimatedFPS(ctx);

    // saveRknnInferenceDetail(ctx);









    delete [] rknnInputs;
    delete [] rknnOutputs;

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        delete [] inbufs[i];
    }
    for (uint32_t i = 0; i < numInputs; ++i)
    {
        delete [] outbufs[i];
    }

    delete [] numInputElems;
    delete [] numOutputElems;

    if (ctx > 0)
    {
        rknn_destroy(ctx);
    }

    delete [] modelbuf;


    return 0;
}