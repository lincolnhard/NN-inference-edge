#pragma once

#include "NvInferPlugin.h"



/// Launcher for argmax kernel
/// \param[in] num_batches Number of batches
/// \param[in] in_dims Input tensor dimensions (without batches)
/// \param[in] data_in Input tensor data
/// \param[in] out_dims Output tensor size
/// \param[out] data_out Output tensor data. Memory must already be allocated
/// \param[in] axis Axis of argmax operation (this dimension will be flattened)
///     0 <= axis < in_dims.nbDims
/// \param[in] type Data type of input and output tensors
/// \param[in] stream Cuda stream to work with
/// \return true on success
bool LauncherArgMax(int const num_batches, nvinfer1::Dims const in_dims,
                   void const *data_in, nvinfer1::Dims const out_dims,
                   void *data_out, int const axis,
                   nvinfer1::DataType const type, cudaStream_t stream);


