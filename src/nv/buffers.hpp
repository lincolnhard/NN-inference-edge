#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <new>          // std::bad_alloc
#include <numeric>      // std::accumulate
#include <functional>   // std::multiplies
#include <iostream>

namespace gallopwave
{



template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT):
        mSize(0), mCapacity(0), mType(type), mBuffer(nullptr)
    {
    }

    GenericBuffer(size_t size, nvinfer1::DataType type):
        mSize(size), mCapacity(size), mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf):
        mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            if (mBuffer != nullptr)
            {
                freeFn(mBuffer);
            }
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    void* data()
    {
        return mBuffer;
    }

    const void* data() const
    {
        return mBuffer;
    }

    size_t size() const
    {
        return mSize;
    }

    size_t nbBytes() const
    {
        size_t elementSize{0};
        switch (mType)
        {
            case nvinfer1::DataType::kINT32:
            case nvinfer1::DataType::kFLOAT:
                elementSize = 4;
                break;
            case nvinfer1::DataType::kHALF:
                elementSize = 2;
                break;
            case nvinfer1::DataType::kINT8:
                elementSize = 1;
                break;
        }
        return this->size() * elementSize;
    }

    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()));
    }

private:
    size_t mSize{0};        // number of elements
    size_t mCapacity{0};    // max number of elements, see resize
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

// The ManagedBuffer class groups together a pair of corresponding device and host buffers.
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

}

#endif