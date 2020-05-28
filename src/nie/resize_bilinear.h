#pragma once

#include <string>
#include <vector>
#include <NvInferPlugin.h>
#include <NvUffParser.h>
#include "resize_plugin.h"



/// Resize tensors with [batch, channels, height, width] size
/// Resize is only for height and width dimensions
/// \note Only last two dimensions of 4D tensor can be resized
/// \note In Tensorflow the tensor is [batch,height,width,channels] (NHWC)
///     but TensorRT only supports NCHW order at the moment
class ResizeBilinear : public ResizePlugin {
   public:
    ResizeBilinear();
    virtual ~ResizeBilinear();

    /// Clone constructor
    /// \param[in] input Dimensions of input tensor
    /// \param[in] size Size of output images (last two dims). size must be 2D
    /// \param[in] type Type of engine (/input data)
    /// \param[in] align_corners Same as tensorflows align_corners for
    ///     resize_bilinear
    ResizeBilinear(nvinfer1::Dims const input_dims,
                   nvinfer1::Dims const output_dims, nvinfer1::Dims const size,
                   nvinfer1::DataType const type, bool const align_corners);

    /// Deserialize constructor
    /// \param[in] data Byte stream from which the plugin should be created
    /// \param[in] length Length of byte stream from which the plugin should be
    ///     created
    ResizeBilinear(void const *data, size_t const length);

    /// \return Plugin name which should be used as 'op="resize_bilinear_TRT"'
    /// in
    ///     graphsurgeon when calling 'create_plugin_node'
    virtual char const *getPluginType() const override;

    /// \return Plugin version
    virtual char const *getPluginVersion() const override;

    /// Run the actual kernel
    /// \param[in] batch_size Size of batch
    /// \param[in] inputs Input tensors
    /// \param[in] outpus Output tensors
    /// \param[in] workspace Workspace (extra memory) for the plugin to work
    ///     with
    /// \param[in] stream Cuda stream which can be used to work with the data
    /// \return 0 is success
    virtual int enqueue(int batch_size, const void *const *inputs,
                        void **outputs, void *workspace,
                        cudaStream_t stream) override;

    /// Clone the plugin
    /// \return New plugin
    virtual nvinfer1::IPluginV2 *clone() const override;
};

class ResizeBilinearPluginCreator : public ResizePluginCreator {
   public:
    ResizeBilinearPluginCreator();

    /// \return Plugin name which should be used as 'op="resize_bilinear_TRT"'
    /// in graphsurgeon when calling 'create_plugin_node'
    virtual char const *getPluginName() const override;

    /// \return Plugin version
    virtual char const *getPluginVersion() const override;

    /// Create plugin with name 'name' using collection of fields
    /// \param[in] name Name of the plugin to create
    /// \param[in] fc Collection of parameters to the plugin
    virtual nvinfer1::IPluginV2 *createPlugin(
        char const *name, nvinfer1::PluginFieldCollection const *fc) override;

    /// Deserialize plugin = create plugin from stream
    /// \param[in] name Name of the plugin which is being deserialized
    /// \param[in] serialData Serialized data from which the plugin should be
    ///     created
    /// \param[in] serialLength Length of serialized data
    /// \return Plugin created from the stream
    virtual nvinfer1::IPluginV2 *deserializePlugin(
        char const *name, void const *serial_data,
        size_t serial_length) override;
};

// Register plugin to plugin registry so that it can be found automatically
REGISTER_TENSORRT_PLUGIN(ResizeBilinearPluginCreator);


