#pragma once

#include <string>
#include <vector>
#include "NvInferPlugin.h"
#include "NvUffParser.h"



/// Resize tensors with [batch, channels, height, width] size
/// Resize is only for height and width dimensions
/// \note Only last two dimensions of 4D tensor can be resized
/// \note In Tensorflow the tensor is [batch,height,width,channels] (NHWC)
///     but TensorRT only supports NCHW order at the moment
class ResizePlugin : public nvinfer1::IPluginV2 {
   public:
    ResizePlugin();
    virtual ~ResizePlugin() override;

    /// Clone constructor
    /// \param[in] input_dims Dimensions of input tensor
    /// \param[in] output_dims Dimensions of output tensor
    /// \param[in] size Size of output images (last two dims). size must be 2D
    /// \param[in] type Type of engine (/input data)
    /// \param[in] align_corners Same as tensorflows align_corners for
    ///     resize_bilinear
    ResizePlugin(nvinfer1::Dims const input_dims,
                 nvinfer1::Dims const output_dims, nvinfer1::Dims const size,
                 nvinfer1::DataType const type, bool const align_corners);

    /// Deserialize constructor
    /// \param[in] data Byte stream from which the plugin should be created
    /// \param[in] length Length of byte stream from which the plugin should be
    ///     created
    ResizePlugin(void const *data, size_t const length);

    /// \return Number of output tensors
    virtual int getNbOutputs() const override;

    /// \param[in] index Index of output tensor
    /// \param[in] inputs Size of input tensors
    /// \param[in] nbInputDims Number of input tensors
    /// \return output dimensions of the plugin
    virtual nvinfer1::Dims getOutputDimensions(int index,
                                               nvinfer1::Dims const *input_dims,
                                               int num_input) override;

    /// \return True if type and format are supported by the plugin (at the same
    ///     time)
    virtual bool supportsFormat(nvinfer1::DataType type,
                                nvinfer1::PluginFormat format) const override;

    /// Called before initialize
    /// \param[in] input_dims Size of input tensors
    /// \param[in] num_inputs Number of input tensors
    /// \param[in] output_dims Size of output tensors
    /// \param[in] num_output Number of output tensors
    /// \param[in] type Type of the engine (type class of input and output data)
    /// \param[in] format Format of the engine (order of data)
    /// \param[in] max_batch_size Maximum number of batches fed to the plugin
    virtual void configureWithFormat(nvinfer1::Dims const *input_dims,
                                     int num_inputs,
                                     nvinfer1::Dims const *output_dims,
                                     int num_output, nvinfer1::DataType type,
                                     nvinfer1::PluginFormat format,
                                     int max_batch_size) override;

    /// Initialize plugin
    /// \return 0 for success
    virtual int initialize() override;

    /// Terminate plugin (deinitialize)
    virtual void terminate() override;

    /// \param[in] max_batch_size Maximum number of batches fed all at once to
    ///     the plugin
    /// \return Needed workspace size in bytes
    virtual size_t getWorkspaceSize(int max_batch_size) const override;

    /// \return Size of plugin serialization in bytes
    virtual size_t getSerializationSize() const override;

    /// Serialize plugin to stream
    /// \param[in] buffer Buffer to which the plugin should be serialized
    /// \note Serialization size should be at most getSerializationSize
    virtual void serialize(void *buffer) const override;

    /// Delete the plugin
    virtual void destroy() override;

    /// Set plugin namespace
    /// \param[in] plugin_namespace Namespace which should be assigned to the
    ///    plugin
    virtual void setPluginNamespace(char const *plugin_namespace) override;

    /// \return Plugin namespace
    virtual char const *getPluginNamespace() const override;

    /// \param[in] dims 2D size [height width] of resized images
    void SetSize(nvinfer1::Dims const dims);

    /// \param[in] align_corners Tensorflow default is false. True should almost
    ///     always be used as it is much more common in other applications.
    void SetAlignCorners(bool const align_corners);

    /// Set shape of input tensor. Used for debugging
    /// \param[in] dims Dimensions of input tensor
    void SetInputShape(nvinfer1::Dims const dims);

    /// Set shape of output tensor. Used for debugging
    /// \param[in] dims Dimensions of output tensor
    void SetOutputShape(nvinfer1::Dims const dims);

    /// \return Shape of output tensor
    nvinfer1::Dims GetOutputShape() const;

   protected:
    /// \return Input dimension size
    nvinfer1::Dims GetInputDims() const;

    /// \return Output dimension size
    nvinfer1::Dims GetOutputSize() const;

    /// \return Align corners
    bool GetAlignCorners() const;

    /// \return Output type
    nvinfer1::DataType GetOutputType() const;

   private:
    // Dimensions of the input (without batches)
    nvinfer1::Dims input_dims_;

    // Dimensions of the output
    nvinfer1::Dims output_dims_;

    // Type of input data
    nvinfer1::DataType type_;

    // If false (default in tensorflow) the interpolation has strange
    // coordinates. Prefer true. Default false
    bool align_corners_;

    // Size of output images [width height]
    nvinfer1::Dims size_;

    // Stuff which is not serialized

    // Namespace of the plugin
    std::string namespace_;
};

class ResizePluginCreator : public nvinfer1::IPluginCreator {
   public:
    ResizePluginCreator();
    ~ResizePluginCreator() override;

    /// \return Field names used by the plugin. For example 'axis'
    virtual nvinfer1::PluginFieldCollection const *getFieldNames() override;

    /// Create plugin with name 'name' using collection of fields
    /// \param[in] name Name of the plugin to create
    /// \param[in] fc Collection of parameters to the plugin
    virtual void createPlugin_initializer(
        ResizePlugin *plugin, char const *name,
        nvinfer1::PluginFieldCollection const *fc);

    /// Set plugin namespace
    /// \param[in] pluginNamespace Namespace which should be assigned to creator
    virtual void setPluginNamespace(char const *plugin_namespace) override;

    /// Get plugin namespace
    /// \return Current creator plugin namespace
    virtual char const *getPluginNamespace() const override;

   private:
    // Container which holds the fields which the plugin needs
    std::vector<nvinfer1::PluginField> field_collection_static_;

    // Pointers to the fields which the plugin needs
    nvinfer1::PluginFieldCollection field_collection_;

    // Namespace of the creator/plugin
    std::string namespace_;
};


