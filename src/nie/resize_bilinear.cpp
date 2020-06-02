#include "resize_bilinear.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include "resize_bilinear_kernel.h"


using namespace nvinfer1;

namespace {
char const *ResizeBilinear_PLUGIN_VERSION{"1"};
char const *ResizeBilinear_PLUGIN_NAME{"resize_bilinear_TRT"};
}  // namespace

ResizeBilinear::ResizeBilinear() {}

ResizeBilinear::~ResizeBilinear() {}

ResizeBilinear::ResizeBilinear(Dims const input_dims, Dims const output_dims,
                               Dims const size, DataType const type,
                               bool const align_corners)
    : ResizePlugin(input_dims, output_dims, size, type, align_corners) {}

ResizeBilinear::ResizeBilinear(void const *data, size_t const length)
    : ResizePlugin(data, length) {}

char const *ResizeBilinear::getPluginType() const {
    // Must match getPluginName
    return ResizeBilinear_PLUGIN_NAME;
}

char const *ResizeBilinear::getPluginVersion() const {
    return ResizeBilinear_PLUGIN_VERSION;
}

int ResizeBilinear::enqueue(int batch_size, const void *const *inputs,
                            void **outputs, void *workspace,
                            cudaStream_t stream) {
    // Run the kernel
    bool const success = LauncherResizeBilinear(
        batch_size, GetInputDims(), inputs[0], GetOutputSize(), GetOutputType(),
        GetAlignCorners(), stream, outputs[0]);

    return success ? 0 : -1;
}

IPluginV2 *ResizeBilinear::clone() const {
    return new ResizeBilinear(GetInputDims(), GetOutputShape(), GetOutputSize(),
                              GetOutputType(), GetAlignCorners());
}

ResizeBilinearPluginCreator::ResizeBilinearPluginCreator()
    : ResizePluginCreator() {}

char const *ResizeBilinearPluginCreator::getPluginName() const {
    return ResizeBilinear_PLUGIN_NAME;
}

char const *ResizeBilinearPluginCreator::getPluginVersion() const {
    return ResizeBilinear_PLUGIN_VERSION;
}

IPluginV2 *ResizeBilinearPluginCreator::createPlugin(
    char const *name, const PluginFieldCollection *fc) {
    auto plugin = std::make_unique<ResizeBilinear>();
    createPlugin_initializer(plugin.get(), name, fc);
    return plugin.release();
}

IPluginV2 *ResizeBilinearPluginCreator::deserializePlugin(
    char const *name, void const *serial_data, size_t serial_length) {
    return new ResizeBilinear(serial_data, serial_length);
}
