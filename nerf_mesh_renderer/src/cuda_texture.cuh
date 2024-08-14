#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "common.h"

class CudaTexture {
public:
    CudaTexture(size_t width, size_t height, std::vector<uint8_t> bytes, bool bSrgb);
    virtual ~CudaTexture();

    [[nodiscard]] inline cudaTextureObject_t getCudaTextureObject() const { return _cuTexture; }

private:
    cudaTextureObject_t _cuTexture = 0u;
    cudaArray_t _cuArray = nullptr;
};