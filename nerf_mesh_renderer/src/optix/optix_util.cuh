#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

/** Convert GLM's vec3 to CUDA's built-in float3 type. Needed by some OptiX functions. */
__forceinline__ __device__ float3 toFloat3(const glm::vec3& v) {
    return { v.x, v.y, v.z };
}

/** Convert CUDA's built-in float3 type to GLM's vec3. */
__forceinline__ __device__ glm::vec3 toVec3(const float3& f) {
    return { f.x, f.y, f.z };
}

__forceinline__ __device__ glm::vec4 toVec4(const float4& f) {
    return { f.x, f.y, f.z, f.w };
}

// Below are functions from the OptiX SDK (slightly adjusted to use GLM). May or may not need them in the future.

__forceinline__ __device__ glm::vec3 toSrgb(const glm::vec3& c ){
    float invGamma = 1.0f / 2.4f;
    const glm::vec3 powed = glm::vec3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
    return { c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
             c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
             c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f };
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x) {
    x = glm::clamp(x, 0.0f, 1.0f);
    enum { N = (1 << 8) - 1, NP1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)NP1), (unsigned int)N);
}

__forceinline__ __device__ Eigen::Array4f makeColorEigen(const glm::vec3& c){
    // First apply gamma, then convert to unsigned char.
    const glm::vec3 srgb = toSrgb(glm::clamp(c, 0.0f, 1.0f));
    return Eigen::Array4f(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}

__forceinline__ __device__ uchar4 makeColor(const glm::vec3& c){
    // First apply gamma, then convert to unsigned char.
    const glm::vec3 srgb = toSrgb(glm::clamp(c, 0.0f, 1.0f));
    return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}

__forceinline__ __device__ uchar4 makeColor(float r, float g, float b){
    return makeColor(glm::vec3(r, g, b));
}
