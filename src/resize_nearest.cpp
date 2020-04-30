#include "resize_nearest.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include "resize_nearest_kernel.h"
#include "trt_common.h"


using namespace nvinfer1;

namespace {
char const *ResizeNearest_PLUGIN_VERSION{"1"};
char const *ResizeNearest_PLUGIN_NAME{"resize_nearest_neighbor_TRT"};
}  // namespace

ResizeNearest::ResizeNearest() {}

ResizeNearest::~ResizeNearest() {}

ResizeNearest::ResizeNearest(Dims const input_dims, Dims const output_dims,
                             Dims const size, DataType const type,
                             bool const align_corners)
    : ResizePlugin(input_dims, output_dims, size, type, align_corners) {}

ResizeNearest::ResizeNearest(void const *data, size_t const length)
    : ResizePlugin(data, length) {}

char const *ResizeNearest::getPluginType() const {
    // Must match getPluginName
    return ResizeNearest_PLUGIN_NAME;
}

char const *ResizeNearest::getPluginVersion() const {
    return ResizeNearest_PLUGIN_VERSION;
}

int ResizeNearest::enqueue(int batch_size, const void *const *inputs,
                           void **outputs, void *workspace,
                           cudaStream_t stream) {
    // Run the kernel
    bool const success = LauncherResizeNearest(
        batch_size, GetInputDims(), inputs[0], GetOutputSize(), GetOutputType(),
        GetAlignCorners(), stream, outputs[0]);

    return success ? 0 : -1;
}

IPluginV2 *ResizeNearest::clone() const {
    return new ResizeNearest(GetInputDims(), GetOutputShape(), GetOutputSize(),
                             GetOutputType(), GetAlignCorners());
}

ResizeNearestPluginCreator::ResizeNearestPluginCreator()
    : ResizePluginCreator() {}

char const *ResizeNearestPluginCreator::getPluginName() const {
    return ResizeNearest_PLUGIN_NAME;
}

char const *ResizeNearestPluginCreator::getPluginVersion() const {
    return ResizeNearest_PLUGIN_VERSION;
}

IPluginV2 *ResizeNearestPluginCreator::createPlugin(
    char const *name, const PluginFieldCollection *fc) {
    auto plugin = std::make_unique<ResizeNearest>();
    createPlugin_initializer(plugin.get(), name, fc);
    return plugin.release();
}

IPluginV2 *ResizeNearestPluginCreator::deserializePlugin(
    char const *name, void const *serial_data, size_t serial_length) {
    return new ResizeNearest(serial_data, serial_length);
}
