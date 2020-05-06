#include "reshape.h"

#include <cassert>
#include <cstring>
#include <memory>
#include "trt_common.h"



using namespace nvinfer1;

namespace {
char const *RESHAPE_PLUGIN_VERSION{"1"};
char const *RESHAPE_PLUGIN_NAME{"reshape_TRT"};
}  // namespace

Reshape::Reshape() {}

Reshape::~Reshape() {}

Reshape::Reshape(const Dims input_dims, const Dims output_dims,
                 const DataType type)
    : input_dims_(input_dims), output_dims_(output_dims), type_(type) {}

Reshape::Reshape(const void *data, const size_t length) {
    POSSIBLY_UNUSED_VARIABLE(length);

    char const *backup = reinterpret_cast<char const *>(data);
    char const *current = backup;

    input_dims_ = ReadFromBuffer<Dims>(current);
    output_dims_ = ReadFromBuffer<Dims>(current);
    type_ = ReadFromBuffer<nvinfer1::DataType>(current);

    // Make sure correct number of bytes were copied
    assert(current == backup + length);
}

const char *Reshape::getPluginType() const { return RESHAPE_PLUGIN_NAME; }

const char *Reshape::getPluginVersion() const { return RESHAPE_PLUGIN_VERSION; }

int Reshape::getNbOutputs() const { return 1; }

Dims Reshape::getOutputDimensions(int index, const Dims *input_dims,
                                  int num_inputs) {
    return output_dims_;
}

bool Reshape::supportsFormat(DataType type, PluginFormat format) const {
    return true;
}

void Reshape::configureWithFormat(const Dims *input_dims, int num_inputs,
                                  const Dims *output_dims, int num_outputs,
                                  DataType type, PluginFormat format,
                                  int max_batch_size) {
    input_dims_ = input_dims[0];
    output_dims_ = output_dims[0];
    type_ = type;
}

int Reshape::initialize() { return 0; }

void Reshape::terminate() {}

size_t Reshape::getWorkspaceSize(int max_batch_size) const { return 0; }

int Reshape::enqueue(int batch_size, const void *const *inputs, void **outputs,
                     void *workspace, cudaStream_t stream) {
    auto bytes = Volume(output_dims_) * batch_size;
    if (type_ == DataType::kHALF) {
        bytes *= 2;
    } else if (type_ == DataType::kINT32 || type_ == DataType::kFLOAT) {
        bytes *= 4;
    }

    GPU_CHECK_ASSERT(
        cudaMemcpyAsync(outputs[0], inputs[0], bytes, cudaMemcpyDeviceToDevice, stream));

    return 0;
}

size_t Reshape::getSerializationSize() const {
    return sizeof(Dims) * 2 +  // input + output dims
            sizeof(DataType);  // type_
}

void Reshape::serialize(void *buffer) const
{
    char *backup = reinterpret_cast<char *>(buffer);
    char *current = backup;
    WriteToBuffer(current, input_dims_);
    WriteToBuffer(current, output_dims_);
    WriteToBuffer(current, type_);
    assert(current = backup + getSerializationSize());
}

void Reshape::destroy()
{
    delete this;
}

IPluginV2 *Reshape::clone() const
{
    return new Reshape(input_dims_, output_dims_, type_);
}

void Reshape::setPluginNamespace(const char *plugin_namespace)
{
    namespace_ = plugin_namespace;
}

const char *Reshape::getPluginNamespace() const
{
    return namespace_.c_str();
}

void Reshape::SetInputShape(const Dims input_dims)
{
    input_dims_ = input_dims;
}

void Reshape::SetOutputShape(const Dims output_dims)
{
    output_dims_ = output_dims;
}

PluginFieldCollection ReshapePluginCreator::field_collection_{};
std::vector<PluginField> ReshapePluginCreator::field_collection_static_;

ReshapePluginCreator::ReshapePluginCreator()
{
    // Initialize static member variables if not yet initialized
    if (field_collection_static_.empty()) {
        field_collection_static_.emplace_back(
            PluginField("input_shape", nullptr, PluginFieldType::kDIMS, 1));
        field_collection_static_.emplace_back(
            PluginField("output_shape", nullptr, PluginFieldType::kDIMS, 1));

        field_collection_.nbFields = field_collection_static_.size();
        field_collection_.fields = field_collection_static_.data();
    }
}

const char *ReshapePluginCreator::getPluginName() const
{
    return RESHAPE_PLUGIN_NAME;
}

const char *ReshapePluginCreator::getPluginVersion() const
{
    return RESHAPE_PLUGIN_VERSION;
}

const PluginFieldCollection *ReshapePluginCreator::getFieldNames()
{
    return &field_collection_;
}

IPluginV2 *ReshapePluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    auto plugin = std::make_unique<Reshape>();
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        char const *attrName = fields[i].name;

        if (!strcmp(attrName, "input_shape")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            Dims dims{0};
            dims.nbDims = fields[i].length;
            auto data = static_cast<int32_t const *>(fields[i].data);
            for (int dim = 0; dim < dims.nbDims; ++dim) {
                dims.d[dim] = data[dim];
            }
            plugin->SetInputShape(dims);
        }

        if (!strcmp(attrName, "output_shape")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            Dims dims{0};
            dims.nbDims = fields[i].length;
            auto data = static_cast<int32_t const *>(fields[i].data);
            for (int dim = 0; dim < dims.nbDims; ++dim) {
                dims.d[dim] = data[dim];
            }
            plugin->SetOutputShape(dims);
        }

    }
    return plugin.release();
}

IPluginV2 *ReshapePluginCreator::deserializePlugin(const char *name, const void *serial_data, size_t serialLength)
{
    return new Reshape(serial_data, serialLength);
}

void ReshapePluginCreator::setPluginNamespace(const char *plugin_namespace)
{
    namespace_ = plugin_namespace;
}

const char *ReshapePluginCreator::getPluginNamespace() const
{
    return namespace_.c_str();
}
