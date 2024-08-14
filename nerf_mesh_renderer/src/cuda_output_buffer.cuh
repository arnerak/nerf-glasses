//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <glad/gl.h> // Needs to be included before gl_interop.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <spdlog/spdlog.h>

#include <cstdint>
#include <vector>

#include "common.h"

enum class BufferImageFormat {
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer {
    void* data = nullptr;
    uint32_t width = 0u;
    uint32_t height = 0u;
    BufferImageFormat pixelFormat = {};

    // TODO: convert this to a general pixelFormatSize(BufferImageFormat) function (as is in the SDK)?
    [[nodiscard]] size_t pixelFormatSize() const {
        switch(pixelFormat) {
            case BufferImageFormat::UNSIGNED_BYTE4:
                return sizeof(char) * 4;
            case BufferImageFormat::FLOAT3:
                return sizeof(float) * 3;
            case BufferImageFormat::FLOAT4:
                return sizeof(float) * 4;
            default:
                throw std::runtime_error("ImageBuffer::pixelFormatSize: Unrecognized buffer format.");
        }
    }
};

struct Texture {
    cudaArray_t array;
    cudaTextureObject_t texture;
};

// Note: removed the CUDA_P2P type which is for directly connected GPUs (e.g. via NVLINK). We won't need that.
enum class CudaOutputBufferType {
    CUDA_DEVICE = 0,    // Not preferred, typically slower than ZERO_COPY.
    GL_INTEROP = 1,     // Single device only; preferred for single device.
    ZERO_COPY = 2,      // General case; preferred for multi-GPU if not fully NVLINK connected.
};

template<typename PIXEL_FORMAT>
class CudaOutputBuffer {
public:
    CudaOutputBuffer(CudaOutputBufferType type, uint32_t width, uint32_t height);
    ~CudaOutputBuffer();

    void resize(uint32_t width, uint32_t height);

    [[nodiscard]] uint32_t width() const;
    [[nodiscard]] uint32_t height() const;
    [[nodiscard]] size_t bufferSize() const;

    void setDevice(uint32_t deviceIndex);
    void setStream(CUstream stream);

    // Allocate or update device pointer as necessary for CUDA access.
    PIXEL_FORMAT* map();
    void unmap();

    // Get output buffer.
    uint32_t getPbo();
    void deletePbo();
    PIXEL_FORMAT* getHostPointer();

private:
    CudaOutputBufferType _type;
    uint32_t _width = 0u;
    uint32_t _height = 0u;

    cudaGraphicsResource* _cudaGraphicsResource = nullptr;
    uint32_t _pbo = 0u;
    PIXEL_FORMAT* _devicePixels = nullptr;
    PIXEL_FORMAT* _hostZeroCopyPixels = nullptr;
    std::vector<PIXEL_FORMAT> _hostPixels;

    CUstream _stream = nullptr;
    uint32_t _deviceIndex = 0u;

    void makeCurrent();
};

// == Implementation ===================================================================================================

template<typename PIXEL_FORMAT>
CudaOutputBuffer<PIXEL_FORMAT>::CudaOutputBuffer(CudaOutputBufferType type, uint32_t width, uint32_t height)
        : _type(type)
{
    // Output dimensions must be at least 1 in both x and y to avoid an error with cudaMalloc.
    common::ensureMinimumSize(width, height);

    // If using GL Interop, expect that the active device is also the display device.
    if(type == CudaOutputBufferType::GL_INTEROP) {
        int currentDevice;
        CUDA_CHECK(cudaGetDevice(&currentDevice));

        int isDisplayDevice;
        CUDA_CHECK(cudaDeviceGetAttribute(&isDisplayDevice, cudaDevAttrKernelExecTimeout, currentDevice));

        if(!isDisplayDevice) {
            throw std::runtime_error(
                    "GL interop is only available on display device, please use display device for optimal "
                    "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
                    "degraded performance."
            );
        }
    }

    resize(width, height);
}

template<typename PIXEL_FORMAT>
CudaOutputBuffer<PIXEL_FORMAT>::~CudaOutputBuffer() {
    try {
        makeCurrent();

        switch(_type) {
            case CudaOutputBufferType::CUDA_DEVICE:
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_devicePixels)));
                break;
            case CudaOutputBufferType::ZERO_COPY:
                CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(_hostZeroCopyPixels)));
                break;
            case CudaOutputBufferType::GL_INTEROP:
                CUDA_CHECK(cudaGraphicsUnregisterResource(_cudaGraphicsResource));
                break;
        }

        if(_pbo != 0u) {
            GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u));
            GL_CHECK(glDeleteBuffers(1, &_pbo));
        }
    }
    catch(std::exception& e) {
        spdlog::error("~CudaOutputBuffer() caught an exception: {}", e.what());
    }
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::resize(uint32_t width, uint32_t height) {
    common::ensureMinimumSize(width, height);

    if(_width == width && _height == height) {
        return;
    }

    _width = width;
    _height = height;

    makeCurrent();

    const size_t bufferSize = this->bufferSize();

    if(_type == CudaOutputBufferType::CUDA_DEVICE) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_devicePixels)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_devicePixels), bufferSize));
    }

    if(_type == CudaOutputBufferType::GL_INTEROP) {
        GL_CHECK(glGenBuffers(1, &_pbo));
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo));
        GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u));

        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&_cudaGraphicsResource, _pbo, cudaGraphicsMapFlagsWriteDiscard));
    }

    if(_type == CudaOutputBufferType::ZERO_COPY) {
        CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(_hostZeroCopyPixels)));
        CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&_hostZeroCopyPixels),
                bufferSize,
                cudaHostAllocPortable | cudaHostAllocMapped
        ));
        CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&_devicePixels),
                reinterpret_cast<void*>(_hostZeroCopyPixels),
                0u  // Flags; must be 0 for now as per documentation.
        ));
    }

    if(_type != CudaOutputBufferType::GL_INTEROP && _pbo != 0u) {
        // Discard the contents of the PBO by calling glBufferData() with appropriate buffer size and nullptr data.
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo));
        GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u));
    }

    if(!_hostPixels.empty()) {
        _hostPixels.resize(_width * _height);
    }
}

template<typename PIXEL_FORMAT>
uint32_t CudaOutputBuffer<PIXEL_FORMAT>::width() const {
    return _width;
}

template<typename PIXEL_FORMAT>
uint32_t CudaOutputBuffer<PIXEL_FORMAT>::height() const {
    return _height;
}

template<typename PIXEL_FORMAT>
size_t CudaOutputBuffer<PIXEL_FORMAT>::bufferSize() const {
    return _width * _height * sizeof(PIXEL_FORMAT);
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::setDevice(uint32_t deviceIndex) {
    _deviceIndex = deviceIndex;
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::setStream(CUstream stream) {
    _stream = stream;
}

template<typename PIXEL_FORMAT>
PIXEL_FORMAT* CudaOutputBuffer<PIXEL_FORMAT>::map() {
    if(_type == CudaOutputBufferType::GL_INTEROP) {
        makeCurrent();

        size_t bufferSize = 0u;
        CUDA_CHECK(cudaGraphicsMapResources(1, &_cudaGraphicsResource, _stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&_devicePixels),
                &bufferSize,
                _cudaGraphicsResource
        ));
    }

    return _devicePixels;
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::unmap() {
    makeCurrent();

    if(_type == CudaOutputBufferType::GL_INTEROP) {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &_cudaGraphicsResource, _stream));
    }
    else {
        CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
}

template<typename PIXEL_FORMAT>
uint32_t CudaOutputBuffer<PIXEL_FORMAT>::getPbo() {
    if(_pbo == 0u) {
        GL_CHECK(glGenBuffers(1, &_pbo));
    }

    const size_t bufferSize = this->bufferSize();

    if(_type == CudaOutputBufferType::CUDA_DEVICE) {
        // A host buffer is needed to act as a way-station.
        if(_hostPixels.empty()) {
            _hostPixels.resize(_width * _height);
        }

        makeCurrent();

        CUDA_CHECK(cudaMemcpy(
                static_cast<void*>(_hostPixels.data()),
                _devicePixels,
                bufferSize,
                cudaMemcpyDeviceToHost
        ));

        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo));
        GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, static_cast<void*>(_hostPixels.data()), GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u));
    }
    else if(_type == CudaOutputBufferType::ZERO_COPY) {
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo));
        GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, static_cast<void*>(_hostZeroCopyPixels), GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u));
    }
    else {
        // GL_INTEROP: nothing needed.
    }

    return _pbo;
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::deletePbo() {
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u));
    GL_CHECK(glDeleteBuffers(1, &_pbo));
    _pbo = 0u;
}

template<typename PIXEL_FORMAT>
PIXEL_FORMAT* CudaOutputBuffer<PIXEL_FORMAT>::getHostPointer() {
    if(_type == CudaOutputBufferType::CUDA_DEVICE || _type == CudaOutputBufferType::GL_INTEROP) {
        _hostPixels.resize(_width * _height);

        makeCurrent();
        CUDA_CHECK(cudaMemcpy(static_cast<void*>(_hostPixels.data()), map(), bufferSize(), cudaMemcpyDeviceToHost));
        unmap();

        return _hostPixels.data();
    }
    else {
        return _hostZeroCopyPixels;
    }
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::makeCurrent() {
    CUDA_CHECK(cudaSetDevice(_deviceIndex));
}