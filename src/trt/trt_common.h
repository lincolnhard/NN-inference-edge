#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include <numeric>
#include "NvInfer.h"
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

constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}

constexpr long double operator"" _MiB(long double val)
{
    return val * (1 << 20);
}

constexpr long double operator"" _KiB(long double val)
{
    return val * (1 << 10);
}

constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}

constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}

constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

#endif // TENSORRT_COMMON_H