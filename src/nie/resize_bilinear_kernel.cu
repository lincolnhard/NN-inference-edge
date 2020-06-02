#include <cassert>
#include <iostream>
#include "resize_bilinear_kernel.h"
#include "common.h"


using namespace nvinfer1;

/// For checking if types are same
template <typename T, typename U>
struct is_same : std::false_type {};

template <typename T>
__device__ struct is_same<T, T> : std::true_type {};

template <typename T, typename U>
constexpr __device__ bool are_types_same() {
    return is_same<T, U>::value;
}

/// \brief CUDA kernel for calculating bilinear resizing of stacked images
///
/// \see https://en.wikipedia.org/wiki/Bilinear_interpolation
/// \see tensorflow/core/kernels/resize_bilinear_op_gpu.cu.cc
///
/// In this implementation there is one thread per output pixel
///
/// \tparam[in] T type of input tensors
/// \tparam[in] align_corners If true, scaling is (in-1)/(out-i), otherwise
///     in/out. True is normal computer vision / image processing bilinear
///     resize, false is default for TensorFlow
/// \param[in] width_in Width of the original image
/// \param[in] height_in Height of the original image
/// \param[in] in Stacked input images in NHW order
/// \param[in] width_out Width of the resulting image
/// \param[in] height_out Height of the resulting image
/// \param[in] output_volume Volume of the output data
///     (width_out*height_out*layers)
/// \param[out] out Output data
template <typename T>
__global__ void KernelResizeBilinear(int const width_in, int const height_in,
                                     T const *in, int const width_out,
                                     int const height_out,
                                     uint32_t const output_volume,
                                     float const x_scale, float const y_scale,
                                     T *out) {
    // Calculate output pixel location
    uint32_t const idx_out = threadIdx.x + blockIdx.x * blockDim.x;

    // Make sure we do not over index mem
    if (idx_out > output_volume) {
        // This thread does not contribute
        return;
    }

    // These are whole integers, but we need them as floats
    float const out_x = static_cast<float>(idx_out % width_out);
    // Note that out_y is actually index in the whole block of images (across
    // layers)
    float const out_y = static_cast<float>(idx_out / width_out);

    // Y-index in 2D image
    float const in_y_img = (static_cast<int>(out_y) % height_out) * y_scale;

    // Input y is calculate little bit too complicated because of block shape
    // and thread indexing
    uint32_t const layer = out_y / height_out;

    // Y-index in the whole 'block' of images
    float const in_y = in_y_img + layer * height_in;

    // Calculate input pixel location (in floats)
    // Align corners is used tensorflows image_resizer_state.h to calculate the
    // scaling
    float const in_x = out_x * x_scale;

    // Convert back to integer. These indeces are the top- and left pixel
    // coordinates. Note use of floorf which is required to get same
    // results as tensorflow. Do not try to optimize it away
    uint32_t const in_xd0 = static_cast<uint32_t>(floorf(in_x));
    uint32_t const in_yd0 = static_cast<uint32_t>(floorf(in_y));

    // Calculate bottom and right pixel coordinates
    // For the last row and column we want to make sure that we do not overindex
    uint32_t const in_xd1 = (in_x < width_in - 1) ? ceilf(in_x) : width_in - 1;
    uint32_t const in_yd1 =
        (in_y_img < height_in - 1) ? ceilf(in_y) : height_in * (layer + 1) - 1;

    // Linear interpolation coefficients
    float const x_lerp = in_x - in_xd0;

    // 4 corner pixels
    // We could try to use __ldg if it improves performance
    // on some devices, but it seems to have same performance on RTX 2080.
    T const py0x0 = in[in_yd0 * width_in + in_xd0];
    T const py1x0 = in[in_yd1 * width_in + in_xd0];
    T const py0x1 = in[in_yd0 * width_in + in_xd1];
    T const py1x1 = in[in_yd1 * width_in + in_xd1];

    float const y_lerp = in_y - in_yd0;

    // Interpolate top and bottom in x-axis
    float const top =
        py0x0 +
        (static_cast<float>(py0x1) - static_cast<float>(py0x0)) * x_lerp;
    float const bot =
        py1x0 +
        (static_cast<float>(py1x1) - static_cast<float>(py1x0)) * x_lerp;

    // Finalize by y-axis interpolation
    if (are_types_same<int8_t, T>()) {
        out[idx_out] =
            fmaxf(fminf(roundf(top + (bot - top) * y_lerp), 127.0f), -128.0f);
    } else {
        out[idx_out] = top + (bot - top) * y_lerp;
    }
}

bool LauncherResizeBilinear(int const num_batches, Dims const in_dims,
                                void const *data_in, Dims const size,
                                DataType const type, bool const align_corners,
                                cudaStream_t stream, void *data_out) {
    assert(size.nbDims == 2);
    assert(in_dims.nbDims >= 2);

    // Calculate output tensor size
    Dims out_dims = in_dims;
    out_dims.d[out_dims.nbDims - 2] = size.d[0];
    out_dims.d[out_dims.nbDims - 1] = size.d[1];

    int const width_in = in_dims.d[in_dims.nbDims - 1];
    int const height_in = in_dims.d[in_dims.nbDims - 2];
    int const width_out = size.d[1];
    int const height_out = size.d[0];
    int const num_outputs = Volume(out_dims) * num_batches;

    // Kernel configuration
    int const threads_per_block = 128;
    int const blocks =
        (num_outputs + threads_per_block - 1) / threads_per_block;
    int const shared_mem = 0;

    // Calculate input pixel location (in floats)
    // Align corners is used tensorflows image_resizer_state.h to calculate the
    // scaling
    float const x_scale = align_corners
                              ? (width_in - 1) / static_cast<float>(width_out - 1)
                              : width_in / static_cast<float>(width_out);
    float const y_scale = align_corners
                              ? (height_in - 1) / static_cast<float>(height_out - 1)
                              : height_in / static_cast<float>(height_out);

    if (DataType::kFLOAT == type) {
        KernelResizeBilinear<float>
            <<<blocks, threads_per_block, shared_mem, stream>>>(
                width_in, height_in, static_cast<float const *>(data_in),
                width_out, height_out, num_outputs, x_scale, y_scale,
                static_cast<float *>(data_out));
    } else if (DataType::kINT8 == type) {
        KernelResizeBilinear<int8_t>
            <<<blocks, threads_per_block, shared_mem, stream>>>(
                width_in, height_in, static_cast<int8_t const *>(data_in),
                width_out, height_out, num_outputs, x_scale, y_scale,
                static_cast<int8_t *>(data_out));
    } else {
        std::cerr << "LauncherResizeBilinear:Unsupported data type" << std::endl;
        return false;
    }

    auto err = cudaStreamSynchronize(stream);
    if (cudaSuccess != err) {
        std::cerr << "LauncherResizeBilinear:Kernel launch failed: "
                  << cudaGetErrorName(err) << std::endl;
        return false;
    }

    return true;
}
