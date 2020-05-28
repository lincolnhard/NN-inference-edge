#include "resize_plugin.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include "common.h"


using namespace nvinfer1;

ResizePlugin::ResizePlugin() : type_(DataType::kFLOAT), align_corners_(false) {
    memset(&input_dims_, 0, sizeof(Dims));
    memset(&size_, 0, sizeof(Dims));
    size_.nbDims = 2;
}

ResizePlugin::~ResizePlugin() {}

ResizePlugin::ResizePlugin(Dims const input_dims, Dims const output_dims,
                           Dims const size, DataType const type,
                           bool const align_corners)
    : input_dims_(input_dims),
      output_dims_(output_dims),
      type_(type),
      align_corners_(align_corners),
      size_(size) {}

ResizePlugin::ResizePlugin(void const *data, size_t const length) {
    POSSIBLY_UNUSED_VARIABLE(length);

    char const *backup = reinterpret_cast<char const *>(data);
    auto current = backup;

    SetInputShape(ReadFromBuffer<Dims>(current));
    SetOutputShape(ReadFromBuffer<Dims>(current));
    type_ = ReadFromBuffer<DataType>(current);
    align_corners_ = ReadFromBuffer<bool>(current);
    SetSize(ReadFromBuffer<Dims>(current));

    // Make sure correct number of bytes were copied
    assert(current == backup + length);
}

void ResizePlugin::serialize(void *buffer) const {
    char *backup = reinterpret_cast<char *>(buffer);
    auto current = backup;
    WriteToBuffer(current, input_dims_);
    WriteToBuffer(current, output_dims_);
    WriteToBuffer(current, type_);
    WriteToBuffer(current, align_corners_);
    WriteToBuffer(current, size_);
    assert(current == backup + getSerializationSize());
}

int ResizePlugin::getNbOutputs() const { return 1; }

Dims ResizePlugin::getOutputDimensions(int index, Dims const *input_dims,
                                       int num_input) {
    // Only one input tensor is supported
    assert(num_input == 1);
    // Only one output tensor is supported
    assert(index == 0);

    auto in = input_dims[0];

    // Construct output size by copying size_ to last two dims of out
    Dims out = in;
    out.d[out.nbDims - 2] = size_.d[0];
    out.d[out.nbDims - 1] = size_.d[1];

    return out;
}

bool ResizePlugin::supportsFormat(DataType type, PluginFormat format) const {
    // Support for kINT8 is experimental.
    return (type == DataType::kFLOAT || type == DataType::kINT8) &&
           format == PluginFormat::kNCHW;
}

void ResizePlugin::configureWithFormat(Dims const *input_dims, int num_inputs,
                                       Dims const *output_dims, int num_output,
                                       DataType type, PluginFormat format,
                                       int max_batch_size) {
    assert(num_inputs == 1);
    assert(num_output == 1);
    assert(format == PluginFormat::kNCHW);

    input_dims_ = input_dims[0];

    // size_ will be last two dims of output
    auto const out = output_dims[0];
    size_.d[0] = out.d[out.nbDims - 2];
    size_.d[1] = out.d[out.nbDims - 1];

    type_ = type;
}

int ResizePlugin::initialize() {
    // Do nothing at the moment
    return 0;
}

void ResizePlugin::terminate() {
    // Do nothing at the moment
}

size_t ResizePlugin::getWorkspaceSize(int max_batch_size) const {
    // We do not se scratch space
    return 0;
}

size_t ResizePlugin::getSerializationSize() const {
    return sizeof(Dims) * 3 +                // input + output + size dims
           sizeof(DataType) + sizeof(bool);  // align_corners
}

void ResizePlugin::destroy() { delete this; }

void ResizePlugin::setPluginNamespace(const char *plugin_namespace) {
    namespace_ = plugin_namespace;
}

char const *ResizePlugin::getPluginNamespace() const {
    return namespace_.c_str();
}

void ResizePlugin::SetSize(Dims const size) {
    // Resize must be 2d
    assert(size.nbDims == 2);
    size_ = size;
}

void ResizePlugin::SetAlignCorners(bool const align_corners) {
    align_corners_ = align_corners;
}

void ResizePlugin::SetInputShape(const Dims dims) { input_dims_ = dims; }

void ResizePlugin::SetOutputShape(const Dims dims) { output_dims_ = dims; }

Dims ResizePlugin::GetOutputShape() const { return output_dims_; }

Dims ResizePlugin::GetInputDims() const { return input_dims_; }

Dims ResizePlugin::GetOutputSize() const { return size_; }

bool ResizePlugin::GetAlignCorners() const { return align_corners_; }

DataType ResizePlugin::GetOutputType() const { return type_; }

ResizePluginCreator::ResizePluginCreator() {
    // Initialize static member variables if not yet initialized
    if (field_collection_static_.empty()) {
        field_collection_static_.emplace_back(
            PluginField("size", nullptr, PluginFieldType::kDIMS, 1));
        field_collection_static_.emplace_back(
            PluginField("align_corners", nullptr, PluginFieldType::kINT8, 1));
        field_collection_static_.emplace_back(
            PluginField("input_shape", nullptr, PluginFieldType::kDIMS, 1));
        field_collection_static_.emplace_back(
            PluginField("output_shape", nullptr, PluginFieldType::kDIMS, 1));

        field_collection_.nbFields = field_collection_static_.size();
        field_collection_.fields = field_collection_static_.data();
    }
}

ResizePluginCreator::~ResizePluginCreator() {}

PluginFieldCollection const *ResizePluginCreator::getFieldNames() {
    return &field_collection_;
}

void ResizePluginCreator::createPlugin_initializer(
    ResizePlugin *plugin, char const *name, const PluginFieldCollection *fc) {
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        char const *attrName = fields[i].name;
        if (!strcmp(attrName, "size")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            assert(fields[i].length == 2);
            Dims dims{0};
            dims.nbDims = fields[i].length;
            auto data = static_cast<int32_t const *>(fields[i].data);
            dims.d[0] = data[0];
            dims.d[1] = data[1];
            // ignore type dims.type
            plugin->SetSize(dims);
        }

        if (!strcmp(attrName, "input_shape")) {
            // Input shape is not mandatory
            // It can be used for debugging

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
            // Output shape is not mandatory
            // It can be used for debugging

            assert(fields[i].type == PluginFieldType::kINT32);
            Dims dims{0};
            dims.nbDims = fields[i].length;
            auto data = static_cast<int32_t const *>(fields[i].data);
            for (int dim = 0; dim < dims.nbDims; ++dim) {
                dims.d[dim] = data[dim];
            }
            plugin->SetOutputShape(dims);
        }

        if (!strcmp(attrName, "align_corners")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            auto const align_corners =
                *(static_cast<const int32_t *>(fields[i].data));
            plugin->SetAlignCorners(align_corners);
        }
    }
}

void ResizePluginCreator::setPluginNamespace(char const *plugin_namespace) {
    namespace_ = plugin_namespace;
}

char const *ResizePluginCreator::getPluginNamespace() const {
    return namespace_.c_str();
}
