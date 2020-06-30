#include <cassert>
#include <iostream>
#include "argmax_kernel.h"
#include "common.h"


using namespace nvinfer1;


/// \brief CUDA kernel for computing ArgMax
///
/// Each thread computes one output element.
/// Thread is mapped to first input element from which it
/// starts the argmax computation. After this the thread starts to
/// compare values in input tensor. The compared values are always
/// 'jump'-many elements apart. For example for consecutive elemnts
/// the jump would be 1, and if every other element would be compared
/// it would be 2. There will be num_compared comparisons this way.
///
/// The bigjump and jump variables are used to map the thread from its threadid
/// to the first input element it is going to use.
///
/// \tparam T Type of input and output tensor
/// \param[in] in Input tensor
/// \param[in] jump Number of elements to progress between comparisons
/// \param[in] bigjump Number of elements that needs to be skipped to map to the
///     input from threadid
/// \param[in] num_compared Number of comparisons (+1) made
///     by the thread
/// \param[in] num_outs Number of elements in the output tensor
/// \param[out] out Output tensor
template <typename T>
__global__ void KernelArgMax(T const *in, int const jump, int const bigjump,
                             int const num_compared, int const num_outs,
                             T *out) {
    // Calculate output idx
    int const idx_out = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_out > num_outs) {
        // This thread does no computations
        return;
    }

    // Calculate the start index and get pointer to correct data
    int const idx_in = idx_out / jump * bigjump + idx_out % jump;
    T const *current = &in[idx_in];

    // Get the first value
    int max_idx = 0;
    auto max_val = *current;
    current += jump;

    // Calculate starting idx in input
    for (int i = 1; i < num_compared; ++i) {
        if (max_val < *current) {
            max_val = *current;
            max_idx = i;
        }
        current += jump;
    }

    out[idx_out] = max_idx;
}

bool LauncherArgMax(int const num_batches, Dims const in_dims,
                         void const *data_in, Dims const out_dims,
                         void *data_out, int const in_axis, DataType const type,
                         cudaStream_t stream) {
    // Handle negative axis dimensions
    int axis = in_axis;
    if (axis < 0) {
        // Convert negative axis to positive one
        axis = in_dims.nbDims + axis;
    }

    // Currently supports only one batch
    assert(axis >= 0 && axis < in_dims.nbDims);

    // How many elements to jump forward when calculating argmax (by single
    // thread) The amount is product of dimensions after axis
    int jump = 1;
    for (int i = in_dims.nbDims - 1; i >= 0 && axis < i; --i) {
        jump *= in_dims.d[i];
    }

    // How to transform from output index to input index
    // This is product of dimensions after axis including axis
    int const bigjump = jump * in_dims.d[axis];

    // How many comparison operations are made per thread
    int const num_compared = in_dims.d[axis];

    // Output volume size
    auto const num_outputs = Volume(out_dims) * num_batches;

    // Kernel configuration
    int const threads_per_block = 128;
    int const blocks =
        (num_outputs + threads_per_block - 1) / threads_per_block;
    int const shared_mem = 0;

    if (DataType::kFLOAT == type) {
        KernelArgMax<float><<<blocks, threads_per_block, shared_mem, stream>>>(
            static_cast<const float *>(data_in), jump, bigjump, num_compared,
            num_outputs, static_cast<float *>(data_out));
    } else if (DataType::kINT8 == type) {
        KernelArgMax<int8_t><<<blocks, threads_per_block, shared_mem, stream>>>(
            static_cast<const int8_t *>(data_in), jump, bigjump, num_compared,
            num_outputs, static_cast<int8_t *>(data_out));
    } else {
        std::cerr << "LauncherArgMax:Unsupported data type" << std::endl;
        return false;
    }

    return true;
}

