#pragma once

#include <NvInferRuntimeCommon.h>
#include <numeric>
#include <functional>
#include <cuda_runtime_api.h>
#include <iostream>

#define POSSIBLY_UNUSED_VARIABLE(x) (void)(x)

#define GPU_CHECK_ASSERT(ans) \
    { GpuAssert((ans), __FILE__, __LINE__); }

inline void GpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file
                  << " line: " << line << std::endl;

        if (abort) {
            exit(code);
        }
    }
}

/// \brief Read T from a buffer.
/// \tparam T type of read var
/// \param[in,out] buffer Buffer from which the type is read
/// \return Read variable
template <typename T>
T ReadFromBuffer(char const *&buffer) {
    T val = *reinterpret_cast<T const *>(buffer);
    buffer += sizeof(T);
    return val;
}


/// \brief Writes a T to buffer.
/// \tparam T type of written var
/// \param[in] buffer Buffer to which the val is written
/// \param[in] val Value to be written to buffer
template <typename T>
void WriteToBuffer(char *&buffer, T const &val) {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}


/// \param[in] dims Dims whichs volume should be returned
/// \return Total volume of dimension in cells
inline size_t Volume(nvinfer1::Dims const dims) {
    auto const val = std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                                     std::multiplies<int>());
    return static_cast<size_t>(val);
}