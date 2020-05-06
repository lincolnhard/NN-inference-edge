#pragma once

#include <string>
#include <vector>
#include "NvInferPlugin.h"
#include "NvUffParser.h"



/// Reshape plugin which can be used for debuggin or replacing the TRT default
/// reshape plugin.
class Reshape : public nvinfer1::IPluginV2 {
   public:
    Reshape();
    virtual ~Reshape() override;
    Reshape(nvinfer1::Dims const input_dims, nvinfer1::Dims const output_dims,
           nvinfer1::DataType const type);

    Reshape(void const *data, size_t const length);

    /// \return Plugin name which should be used as 'op="reshape_TRT"' in
    ///     graphsurgeon when calling 'create_plugin_node'
    virtual char const *getPluginType() const override;

    /// \return Plugin version
    virtual char const *getPluginVersion() const override;

    /// \return Number of output tensors
    virtual int getNbOutputs() const override;

    /// \param[in] index Index of output tensor
    /// \param[in] input_dims Size of input tensors
    /// \param[in] num_inputs Number of input tensors
    /// \return output dimensions of the plugin
    virtual nvinfer1::Dims getOutputDimensions(int index,
                                               nvinfer1::Dims const *input_dims,
                                               int num_inputs) override;

    /// \return True if type and format are supported by the plugin (at the same
    /// time)
    virtual bool supportsFormat(nvinfer1::DataType type,
                                nvinfer1::PluginFormat format) const override;

    /// Called before initialize
    /// \param[in] inputDims Size of input tensors
    /// \param[in] num_inputs Number of input tensors
    /// \param[in] output_dims Size of output tensors
    /// \param[in] num_outputs Number of output tensors
    /// \param[in] type Type of the engine (type class of input and output data)
    /// \param[in] format Format of the engine (order of data)
    /// \param[in] max_batch_size Maximum number of batches fed to the plugin
    virtual void configureWithFormat(nvinfer1::Dims const *input_dims,
                                     int num_inputs,
                                     nvinfer1::Dims const *output_dims,
                                     int num_outputs, nvinfer1::DataType type,
                                     nvinfer1::PluginFormat format,
                                     int max_batch_size) override;

    /// Initialize plugin
    /// \return 0 for success
    virtual int initialize() override;

    /// Terminate plugin (deinitialize)
    virtual void terminate() override;

    /// \param[in] max_batch_size Maximum number of batches fed all at once to the
    /// plugin \return Needed workspace size in bytes
    virtual size_t getWorkspaceSize(int max_batch_size) const override;

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

    /// \return Size of plugin serialization in bytes
    virtual size_t getSerializationSize() const override;

    /// Serialize plugin to stream
    /// \param[in] buffer Buffer to which the plugin should be serialized
    /// \note Serialization size should be at most getSerializationSize
    virtual void serialize(void *buffer) const override;

    /// Delete the plugin
    virtual void destroy() override;

    /// Clone the plugin
    /// \return New plugin
    virtual nvinfer1::IPluginV2 *clone() const override;

    /// Set plugin namespace
    /// \param[in] plugin_namespace Namespace which should be assigned to the
    /// plugin
    virtual void setPluginNamespace(const char *plugin_namespace) override;

    /// \return Plugin namespace
    virtual char const *getPluginNamespace() const override;

    /// \param[in] input_dims Input dimensions
    void SetInputShape(nvinfer1::Dims const input_dims);

    /// \param[in] output_dims Output dimensions
    void SetOutputShape(nvinfer1::Dims const output_dims);
 private:
    // Dimensions of the input (without batches)
    nvinfer1::Dims input_dims_;

    // Dimensions of the output
    nvinfer1::Dims output_dims_;

    // Type of input data
    nvinfer1::DataType type_;

    // Stuff which is not serialized

    // Namespace of the plugin
    std::string namespace_;
};

class ReshapePluginCreator : public nvinfer1::IPluginCreator {
   public:
    ReshapePluginCreator();

    /// \return Plugin name which should be used as 'op="reshape_TRT"' in
    ///     graphsurgeon when calling 'create_plugin_node'
    virtual char const *getPluginName() const override;

    /// \return Plugin version
    virtual char const *getPluginVersion() const override;

    /// \return Field names used by the plugin. For example 'axis'
    virtual nvinfer1::PluginFieldCollection const *getFieldNames() override;

    /// Create plugin with name 'name' using collection of fields
    /// \param[in] name Name of the plugin to create
    /// \param[in] fc Collection of parameters to the plugin
    virtual nvinfer1::IPluginV2 *createPlugin(
        char const *name, nvinfer1::PluginFieldCollection const *fc) override;

    /// Deserialize plugin = create plugin from stream
    /// \param[in] name Name of the plugin which is being deserialized
    /// \param[in] serial_data Serialized data from which the plugin should be
    /// created \param[in] serialLength Length of serialized data \return Plugin
    /// created from the stream
    virtual nvinfer1::IPluginV2 *deserializePlugin(
        char const *name, void const *serial_data, size_t serialLength) override;

    /// Set plugin namespace
    /// \param[in] pluginNamespace Namespace which should be assigned to creator
    virtual void setPluginNamespace(char const *plugin_namespace) override;

    /// Get plugin namespace
    /// \return Current creator plugin namespace
    virtual char const *getPluginNamespace() const override;

   private:
    // Container which holds the fields which the plugin needs
    static std::vector<nvinfer1::PluginField> field_collection_static_;

    // Pointers to the fields which the plugin needs
    static nvinfer1::PluginFieldCollection field_collection_;

    // Namespace of the creator/plugin
    std::string namespace_;
};

REGISTER_TENSORRT_PLUGIN(ReshapePluginCreator);


