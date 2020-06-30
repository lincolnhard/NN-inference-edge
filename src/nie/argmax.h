#pragma once

#include <string>
#include <vector>
#include <NvInferPlugin.h>
#include <NvUffParser.h>



class ArgMax : public nvinfer1::IPluginV2 {
   public:
    ArgMax();

    /// Clone constructor
    /// \param[in] axis Direction of argmax operation on the input tensor.
    ///     Basically removes this dimension
    /// \param[in] input_dims Dimensions of input tensor
    /// \param[in] output_dims Dimenions of output tensor
    /// \param[in] type Type of engine (/input data)
    /// \param[in] allow_int8 If true, plugin can use int8 output. This is
    ///     can be required for int8 calculatetions. Default true
    /// \param[in] keepdims If true, input and output tensors will have
    ///     same rank. Otherwise output rank is input rank -1.
    ArgMax(int const axis, nvinfer1::Dims const input_dims,
           nvinfer1::Dims const output_dims, nvinfer1::DataType const type,
           bool const allow_int8, bool const keepdims);

    /// Deserialize constructor
    /// \param[in] data Byte stream from which the plugin should be created
    /// \param[in] length Length of byte stream from which the plugin should be
    /// created
    ArgMax(void const *data, size_t const length);

    /// \return Plugin name which should be used as 'op="argmax_TRT"' in
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

    /// Set axis of argmax. Basically this axis will be removed from the input
    /// data by using function max
    /// \param[in] axis Axis to be removed
    /// \note Valid axis are [-rank(input) - 1, rank(input) - 1)
    void setAxis(int const axis);

    /// Set if int8 is allowed
    void setAllowInt8(bool const allow_int8);

    /// If set to True, output tensor rank will be same as input tensor rank
    void setKeepdims(bool const keepdims) {keepdims_ = keepdims;}
   private:
    // Axis of argmax, default is 0
    int axis_;

    // Dimensions of the input (without batches)
    nvinfer1::Dims input_dims_;

    // Dimensions of the output
    nvinfer1::Dims output_dims_;

    // Type of input data
    nvinfer1::DataType type_;

    // Stuff which is not serialized

    // Namespace of the plugin
    std::string namespace_;

    // If true, int8 output is allowed. Note that this will limit the maximum
    // number for argmax to 255 (we count both negative and positive)
    bool allow_int8_;

    // If true, input and output dimensions will have same rank
    bool keepdims_{false};
};

class ArgMaxPluginCreator : public nvinfer1::IPluginCreator {
   public:
    ArgMaxPluginCreator();

    /// \return Plugin name which should be used as 'op="argmax_TRT"' in
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

// Register plugin to plugin registry so that it can be found automatically
REGISTER_TENSORRT_PLUGIN(ArgMaxPluginCreator);


