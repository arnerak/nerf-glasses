#include "cuda_texture.cuh"

CudaTexture::CudaTexture(size_t width, size_t height, std::vector<uint8_t> bytes, bool bSrgb) {
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    CUDA_CHECK(cudaMallocArray(&_cuArray, &channelDesc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(
            _cuArray,
            0,
            0,
            bytes.data(),
            width * sizeof(uchar4),
            width * sizeof(uchar4),
            height,
            cudaMemcpyHostToDevice
    ));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = _cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.readMode = cudaReadModeNormalizedFloat; // Convert to floating point so that linear filtering can be used.
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = bSrgb ? 1 : 0;

    CUDA_CHECK(cudaCreateTextureObject(&_cuTexture, &resDesc, &texDesc, nullptr));
}

CudaTexture::~CudaTexture() {
    cudaDestroyTextureObject(_cuTexture);
    cudaFreeArray(_cuArray);
}
