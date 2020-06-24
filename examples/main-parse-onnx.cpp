#include <fstream>
#include <map>
#include <string>
#include <numeric>
#include <functional>
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"

#include "onnx/onnx.proto3.pb.h"

static auto LOG = spdlog::stdout_color_mt("MAIN");

void formatString(std::string& str, const std::string& from, const std::string& to)
{
    size_t startpos = 0;
    while ((startpos = str.find(from, startpos)) != std::string::npos)
    {
        str.replace(startpos, from.length(), to);
        startpos += to.length();
    }
}

int main(int argc, char **argv)
{    
    onnx::ModelProto modelproto;
    std::fstream fin("data/nie-mobilenetssd.onnx", std::ios::in | std::ios::binary);
    bool isSuccess = modelproto.ParseFromIstream(&fin);
    if (!isSuccess)
    {
        SLOG_ERROR << "Failed to read the onnx model file" << std::endl;
        return 1;
    }
    fin.close();

    isSuccess = modelproto.has_graph();
    if (!isSuccess)
    {
        SLOG_ERROR << "Can't find network structure" << std::endl;
        return 1;
    }

    std::fstream fout("data/nie-mobilenetssd.nnapiweight", std::ios::out | std::ios::binary);
    onnx::GraphProto graphproto = modelproto.graph();
    SLOG_INFO << "Number of weights: " << graphproto.initializer_size() << std::endl;

    for (int i = 0; i < graphproto.initializer_size(); ++i)
    {
        const onnx::TensorProto& tensorproto = graphproto.initializer(i);
        if (onnx::TensorProto_DataType_FLOAT != tensorproto.data_type())
        {
            SLOG_ERROR << "Unsupported data types, support FP32 only now" << std::endl;
            return 1;
        }

        int tensorSize  = 1;
        for(int j = 0; j < tensorproto.dims_size(); ++j)
        {
		    tensorSize *= tensorproto.dims(j);
	    }

        std::string weightName = tensorproto.name();
        formatString(weightName, "/", "_"); // Avoid conflicts when file creation with filenames that contain "/"
        // SLOG_INFO << weightName << ": " << tensorSize << std::endl;

        const char *dataptr = tensorproto.raw_data().data();
        fout.write(dataptr, tensorSize * sizeof(float));
    }
    fout.close();


    SLOG_INFO << "Number of layers: " << graphproto.node_size() << std::endl;
    for (int layeridx = 0; layeridx < graphproto.node_size(); ++layeridx)
    {
        const onnx::NodeProto nodeproto = graphproto.node(layeridx);
        std::string layerType = nodeproto.op_type();
        SLOG_INFO << layeridx << " " << layerType << std::endl;
        if (layerType == "Conv")
        {
            int pad_top = 0;
            int pad_left = 0;
            int pad_bottom = 0;
            int pad_right = 0;
            int stride_height = 1;
            int stride_width = 1;
            int dilation_height = 1;
            int dilation_width = 1;
            int group = 1;
            for(int attridx = 0; attridx < nodeproto.attribute_size() ; ++attridx)
            {
                const onnx::AttributeProto attributeproto = nodeproto.attribute(attridx);
                std::string attributeName = attributeproto.name();
                if (attributeName == "strides")
                {

                }
            }
        }
    }


    return 0;
}
