#include "argmax.h"

#include <cstring>
#include <cassert>
#include <memory>
#include "argmax_kernel.h"
#include "common.h"


using namespace nvinfer1;

namespace {
char const *ARGMAX_PLUGIN_VERSION{"1"};
char const *ARGMAX_PLUGIN_NAME{"argmax_TRT"};
}  // namespace

ArgMax::ArgMax() : axis_(0), type_(DataType::kFLOAT), allow_int8_(true) {
    memset(&input_dims_, 0, sizeof(Dims));
    memset(&output_dims_, 0, sizeof(Dims));
}

ArgMax::ArgMax(int const axis, Dims const input_dims, Dims const output_dims,
               DataType const type, bool const allow_int8, bool const keepdims)
    : axis_(axis),
      input_dims_(input_dims),
      output_dims_(output_dims),
      type_(type),
      allow_int8_(allow_int8),
      keepdims_(keepdims) {}

ArgMax::ArgMax(void const *data, size_t const length) {
    POSSIBLY_UNUSED_VARIABLE(length);

    char const *backup = reinterpret_cast<char const *>(data);
    char const *current = backup;

    axis_ = ReadFromBuffer<int>(current);
    input_dims_ = ReadFromBuffer<Dims>(current);
    output_dims_ = ReadFromBuffer<Dims>(current);
    type_ = ReadFromBuffer<DataType>(current);
    allow_int8_ = ReadFromBuffer<bool>(current);

    // Make sure correct number of bytes were copied
    assert(current == backup + length);
}

void ArgMax::serialize(void *buffer) const {
    char *backup = reinterpret_cast<char *>(buffer);
    char *current = backup;
    WriteToBuffer(current, axis_);
    WriteToBuffer(current, input_dims_);
    WriteToBuffer(current, output_dims_);
    WriteToBuffer(current, type_);
    WriteToBuffer(current, allow_int8_);
    assert(current = backup + getSerializationSize());
}

char const *ArgMax::getPluginType() const {
    // Must match getPluginName
    return ARGMAX_PLUGIN_NAME;
}

char const *ArgMax::getPluginVersion() const { return ARGMAX_PLUGIN_VERSION; }

int ArgMax::getNbOutputs() const { return 1; }

Dims ArgMax::getOutputDimensions(int index, Dims const *input_dims,
                                 int num_inputs) {
    // Note that this method is not called if engine is created from serialized data
    // Therefore keepdims_ etc do no need to be serialized

    // Only one input tensor is supported
    assert(num_inputs == 1);
    // Only one output tensor is supported
    assert(index == 0);

    auto in = input_dims[0];
    Dims out;
    if (keepdims_) {
        out.nbDims  = in.nbDims;
    } else {
        out.nbDims = in.nbDims - 1;
    }

    int axis_mod = axis_;
    if (axis_mod < 0) {
        // Convert negative axis to positive one
        axis_mod = in.nbDims + axis_mod;
    }

    // Make sure modified axis is in within data limits
    assert(axis_mod >= 0);
    assert(axis_mod < in.nbDims);

    // Copy data from input dims
    // Skip one dim as it is flattened
    int out_idx = 0;
    for (int idx = 0; idx < in.nbDims; ++idx) {
        if (idx == axis_mod) {
            if (keepdims_) {
                out.d[out_idx] = 1;
#if NV_TENSORRT_MAJOR < 6
                out.type[out_idx] = in.type[idx];
#endif
                out_idx++;
            } else {
                // This axis is flattened out by max
            }
            continue;
        }

        // Copy dimension size and type
        out.d[out_idx] = in.d[idx];
#if NV_TENSORRT_MAJOR < 6
        out.type[out_idx] = in.type[idx];
#endif
        out_idx++;
    }

    return out;
}

bool ArgMax::supportsFormat(DataType type, PluginFormat format) const {
    return (type == DataType::kFLOAT ||
            (allow_int8_ && (type == DataType::kINT8))) &&
#if NV_TENSORRT_MAJOR >= 6
           format == PluginFormat::kLINEAR;
#else
           format == PluginFormat::kNCHW;
#endif
}

void ArgMax::configureWithFormat(Dims const *input_dims, int num_inputs,
                                 Dims const *output_dims, int num_outputs,
                                 DataType type, PluginFormat format,
                                 int max_batch_size) {
    assert(num_inputs == 1);
    assert(num_outputs == 1);
#if NV_TENSORRT_MAJOR >= 6
    assert(format == PluginFormat::kLINEAR);
#else
    assert(format == PluginFormat::kNCHW);
#endif

    input_dims_ = input_dims[0];
    output_dims_ = output_dims[0];
    type_ = type;
}

int ArgMax::initialize() {
    // Do nothing at the moment
    return 0;
}

void ArgMax::terminate() {
    // Do nothing at the moment
}

size_t ArgMax::getWorkspaceSize(int max_batch_size) const {
    // We do not se scratch space
    return 0;
}

int ArgMax::enqueue(int batch_size, const void *const *inputs, void **outputs,
                    void *workspace, cudaStream_t stream) {
    // Run the kernel
    bool const success =
        LauncherArgMax(batch_size, input_dims_, inputs[0], output_dims_,
                       outputs[0], axis_, type_, stream);

    return success ? 0 : -1;
}

size_t ArgMax::getSerializationSize() const {
    return sizeof(int) +       // axis_
           sizeof(Dims) * 2 +  // input + output dims
           sizeof(DataType) +  // type_
           sizeof(bool);       // allow_int8_;
}

void ArgMax::destroy() { delete this; }

IPluginV2 *ArgMax::clone() const {
    return new ArgMax(axis_, input_dims_, output_dims_, type_, allow_int8_, keepdims_);
}

void ArgMax::setPluginNamespace(char const *plugin_namespace) {
    namespace_ = plugin_namespace;
}

char const *ArgMax::getPluginNamespace() const { return namespace_.c_str(); }

void ArgMax::setAxis(int const axis) { axis_ = axis; }

void ArgMax::setAllowInt8(bool const allow_int8) { allow_int8_ = allow_int8; }

PluginFieldCollection ArgMaxPluginCreator::field_collection_{};
std::vector<PluginField> ArgMaxPluginCreator::field_collection_static_;

ArgMaxPluginCreator::ArgMaxPluginCreator() {
    // Initialize static member variables if not yet initialized
    if (field_collection_static_.empty()) {
        field_collection_static_.emplace_back(
            PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
        field_collection_static_.emplace_back(
            PluginField("allow_int8", nullptr, PluginFieldType::kINT32, 1));
        field_collection_static_.emplace_back(
            PluginField("keepdims", nullptr, PluginFieldType::kINT32, 1));

        field_collection_.nbFields = field_collection_static_.size();
        field_collection_.fields = field_collection_static_.data();
    }
}

char const *ArgMaxPluginCreator::getPluginName() const {
    return ARGMAX_PLUGIN_NAME;
}

char const *ArgMaxPluginCreator::getPluginVersion() const {
    return ARGMAX_PLUGIN_VERSION;
}

PluginFieldCollection const *ArgMaxPluginCreator::getFieldNames() {
    return &field_collection_;
}

IPluginV2 *ArgMaxPluginCreator::createPlugin(char const *name,
                                             PluginFieldCollection const *fc) {
    auto argmax = std::make_unique<ArgMax>();

    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        char const *attrName = fields[i].name;
        if (!strcmp(attrName, "axis")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            auto const axis = *(static_cast<int const *>(fields[i].data));
            argmax->setAxis(axis);
        } else if (!strcmp(attrName, "allow_int8")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            auto const allow_int8 = *(static_cast<int const *>(fields[i].data));
            argmax->setAllowInt8(allow_int8);
        } else if (!strcmp(attrName, "keepdims")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            auto const keepdims = *(static_cast<int const *>(fields[i].data));
            argmax->setKeepdims(keepdims != 0);
        }
    }

    return argmax.release();
}

IPluginV2 *ArgMaxPluginCreator::deserializePlugin(char const *name,
                                                  void const *serial_data,
                                                  size_t serialLength) {
    return new ArgMax(serial_data, serialLength);
}

void ArgMaxPluginCreator::setPluginNamespace(char const *plugin_namespace) {
    namespace_ = plugin_namespace;
}

char const *ArgMaxPluginCreator::getPluginNamespace() const {
    return namespace_.c_str();
}
