#pragma once

#include <NvInferPlugin.h>



/// Launch ResizeNearest cuda kernel
/// \param[in] num_batches Number of batches
/// \param[in] in_dims Input tensor dimensions
/// \param[in] data_in Input data in GPU
/// \param[in] size Size of output image. 2D (two element) Dims [height width]
/// \param[in] type Type of input and output tensors
/// \param[in] align_corners If true, scaling is (in-1)/(out-i), otherwise
///     in/out. True is normal computer vision / image processing nearest
///     neighbor resize, false is default for TensorFlow
/// \param[in] stream Cuda stream to work with
/// \param[out] data_out Output data of resized images
/// \return True on success
bool LauncherResizeNearest(int const num_batches, nvinfer1::Dims const in_dims,
                           void const *data_in, nvinfer1::Dims const size,
                           nvinfer1::DataType const type,
                           bool const align_corners, cudaStream_t stream,
                           void *data_out);


