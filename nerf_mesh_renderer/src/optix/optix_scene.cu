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

#include "optix_scene.cuh"
#include "optix_util.cuh"

#include <glm/ext/scalar_constants.hpp>

extern "C" {
    __constant__ Params params;
}

// From learnopengl.com.
struct Headlight {
    glm::vec3 position;
    glm::vec3 direction;
    float cutoff;
    float outerCutoff;

    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;

    [[nodiscard]] __device__ float intensity(float theta) const {
        const float eps = cutoff - outerCutoff;
        return glm::clamp((theta - outerCutoff) / eps, 0.0f, 1.0f);
    }

    [[nodiscard]] __device__ float attenuation(float distance) const {
        return 1.0f / (constant + linear * distance + quadratic * (distance * distance));
    }
};

static __forceinline__ __device__ void setPayload(float3 p) {
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ void setPayload(glm::vec3 p) {
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ void computeRay(uint3 index, uint3 dim, glm::vec3& origin, glm::vec3& direction) {
    // [u, v, w] to account for projection.
    const glm::vec3 u = params.camU;
    const glm::vec3 v = params.camV;
    const glm::vec3 w = params.camW;

    // Map thread index to [-1, 1].
    const auto d = glm::vec2(
            2.0f * ((static_cast<float>(index.x) + 0.5f) / static_cast<float>(dim.x)) - 1.0f,
            2.0f * ((static_cast<float>(index.y) + 0.5f) / static_cast<float>(dim.y)) - 1.0f
    );

    origin = params.camEye;
    direction = glm::normalize((d.x * u) + (d.y * v) + w);
}

static __forceinline__ __device__ float computeDepth(const glm::vec3 hitPos) {
    return glm::dot(params.camW, hitPos - params.camEye);
}

/** Gram-Schmidt orthogonalized TBN-matrix. */
static __device__ glm::mat3 computeTbnMatrix(glm::vec3 normal, glm::vec4 tangent, glm::mat3 modelMatrix) {
    glm::vec3 t = glm::normalize(glm::vec3(modelMatrix * glm::vec4(tangent.x, tangent.y, tangent.z, 0.0f)));
    glm::vec3 n = glm::normalize(glm::vec3(modelMatrix * glm::vec4(normal, 0.0f)));
    t = glm::normalize(t - n * glm::dot(t, n));
    glm::vec3 b = glm::cross(n, t) * tangent.w;
    return { t, b, n };
}

static __device__ float dGgx(float dotNH, float roughness) {
    const float a2 = roughness * roughness;
    const float f = (dotNH * a2 - dotNH) * dotNH + 1.0f;
    return a2 / (f * f);
}

static __device__ float gGgx(float dotNL, float dotNV, float roughness) {
    const float a2 = roughness * roughness;
    //const float lambdaV = dotNL * glm::sqrt((-dotNV * a2 + dotNV) * dotNV + a2);
    //const float lambdaL = dotNV * glm::sqrt((-dotNL * a2 + dotNL) * dotNL + a2);
    const float lambdaV = glm::max(0.f, dotNL) / glm::sqrt(a2 + (1.0f - a2) * dotNV * dotNV);
    const float lambdaL = glm::max(0.f, dotNV) / glm::sqrt(a2 + (1.0f - a2) * dotNL * dotNL);
    return 0.5f / (lambdaV + lambdaL + 0.0001f);
}

static __device__ glm::vec3 fSchlick(glm::vec3 f0, float u) {
    const glm::vec3 f90(1.0f);
    return f0 + (f90 - f0) * glm::pow((1.0f - u), 5.0f);
}

extern "C" __global__ void __raygen__rg() {
    // Location within the launch grid.
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint i = idx.y * params.imageWidth + idx.x;

    // Map launch index to a screen location and create a ray from the camera through the screen.
    glm::vec3 rayOrigin;
    glm::vec3 rayDirection;
    computeRay(idx, dim, rayOrigin, rayDirection);

    // Trace the ray against the scene hierarchy.
    uint32_t p0;
    uint32_t p1;
    uint32_t p2;
    uint32_t p3;
    optixTrace(
            params.handle,
            toFloat3(rayOrigin),
            toFloat3(rayDirection),
            0.0f,
            1e16f,
            0.0f,   // rayTime: used for motion blur.
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            0u,      // SBT offset
            1u,      // SBT stride
            0u,      // missSBTIndex
            // Payload...
            p0, // R
            p1, // G
            p2, // B
            p3  // Depth (hitTMax)
    );

    uchar4 pixelColor;
    //if(params.bNoNerfsRendered || __uint_as_float(p3) < params.dDepthBuffer[i]) {
        glm::vec3 result;
        result.x = __uint_as_float(p0);
        result.y = __uint_as_float(p1);
        result.z = __uint_as_float(p2);
        result = glm::clamp(result, 0.f, 1.f);
        result = toSrgb(result);

        params.dFramebuffer[i] = { result.x, result.y, result.z, p3 == -1 ? 0.f : 1.f };
        params.dDepthBuffer[i] = __uint_as_float(p3);
    //}
    //else {
    //    auto fColor = params.dFramebuffer[i];
    //    pixelColor = makeColor(fColor.x(), fColor.y(), fColor.z());
    //}

    // Record results in output raster.
    //params.image[i] = pixelColor;
}

extern "C" __global__ void __miss__ms() {
    const auto missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(missData->backgroundColor);
    optixSetPayload_3(-1/*__float_as_uint(optixGetRayTmax())*/);
}

extern "C" __global__ void __closesthit__ch() {
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, including barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    uint32_t primitiveIndex = optixGetPrimitiveIndex();

    const auto hitGroupData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GeometryData::Mesh mesh = hitGroupData->geometryData.mesh;

    float baryX = barycentrics.x;
    float baryY = barycentrics.y;

    glm::mat4 modelMatrix(1.0f);
    const float4* modelMatrixPtr = optixGetInstanceTransformFromHandle(params.handle);
    if(modelMatrixPtr != nullptr) {
        modelMatrix = glm::mat4(
                toVec4(modelMatrixPtr[0]),
                toVec4(modelMatrixPtr[1]),
                toVec4(modelMatrixPtr[2]),
                toVec4(modelMatrixPtr[3])
        );
    }

    uint16_t indices[3] = {
            mesh.dIndices[primitiveIndex * 3 + 0],
            mesh.dIndices[primitiveIndex * 3 + 1],
            mesh.dIndices[primitiveIndex * 3 + 2]
    };

    glm::vec3 n0 = mesh.dNormals[indices[0]];
    glm::vec3 n1 = mesh.dNormals[indices[1]];
    glm::vec3 n2 = mesh.dNormals[indices[2]];
    glm::vec3 n = baryX * n1 + baryY * n2 + (1.0f - baryX - baryY) * n0;

    glm::vec4 t0 = mesh.dTangents[indices[0]];
    glm::vec4 t1 = mesh.dTangents[indices[1]];
    glm::vec4 t2 = mesh.dTangents[indices[2]];
    glm::vec4 t = baryX * t1 + baryY * t2 + (1.0f - baryX - baryY) * t0;

    glm::vec2 uv0 = mesh.dTexCoords[indices[0]];
    glm::vec2 uv1 = mesh.dTexCoords[indices[1]];
    glm::vec2 uv2 = mesh.dTexCoords[indices[2]];
    glm::vec2 uv = baryX * uv1 + baryY * uv2 + (1.0f - baryX - baryY) * uv0;

    const glm::mat3 tbnMatrix = computeTbnMatrix(n, t, modelMatrix);

    glm::vec4 baseColor = mesh.baseColorFactor;
    if(mesh.baseColorTexture != 0u) {
        baseColor *= toVec4(tex2D<float4>(mesh.baseColorTexture, uv.x, uv.y));
    }

    glm::vec3 emissive = mesh.emissiveFactor;
    if(mesh.emissiveTexture != 0u) {
        const auto tmp = toVec4(tex2D<float4>(mesh.emissiveTexture, uv.x, uv.y));
        emissive *= glm::vec3(tmp.x, tmp.y, tmp.z);
    }

    float metallic = mesh.metallicFactor;
    float roughness = mesh.roughnessFactor;
    if(mesh.metallicRoughnessTexture != 0u) {
        const auto metallicRoughness = toVec4(tex2D<float4>(mesh.metallicRoughnessTexture, uv.x, uv.y));
        metallic *= metallicRoughness.z;      // Metallic is in the B-channel.
        roughness *= metallicRoughness.y;     // Roughness is in the G-channel.
    }

    glm::vec3 normal = n; // Base is geometry normal.
    if(mesh.normalTexture != 0u) {
        const auto tmp = toVec4(tex2D<float4>(mesh.normalTexture, uv.x, uv.y));
        normal = glm::vec3(tmp.x, tmp.y, tmp.z) * 2.0f - glm::vec3(1.0f);
        normal *= glm::vec3(glm::vec2(mesh.normalScale), 1.0f);
        normal = tbnMatrix * normal;
    }
    normal = glm::normalize(*(glm::vec3*)&optixTransformNormalFromObjectToWorldSpace(*(float3*)&normal.x));

    float occlusion = 1.0f;
    if(mesh.occlusionTexture != 0u) {
        occlusion = toVec4(tex2D<float4>(mesh.occlusionTexture, uv.x, uv.y)).x;  // Occlusion is in the R-channel.
        occlusion = 1.0f + mesh.occlusionStrength * (occlusion - 1.0f);
    }

//    setPayload(n);
//    setPayload(glm::vec3(t.x, t.y, t.z));
//    setPayload(emissive);
//    setPayload(glm::vec3(baseColor.x, baseColor.y, baseColor.z));
//    setPayload(glm::vec3(metallic, metallic, metallic));
//    setPayload(glm::vec3(roughness, roughness, roughness));
//    setPayload(normal);
//    setPayload(glm::vec3(occlusion, occlusion, occlusion));

    const float hitT = optixGetRayTmax();
    const glm::vec3 dir = toVec3(optixGetWorldRayDirection());
    const glm::vec3 hitPos = params.camEye + hitT * dir;
    const float depth = hitT;//computeDepth(hitPos);
    optixSetPayload_3(__float_as_uint(depth));

    const glm::vec3 ambient = glm::vec3(baseColor) * .2f * occlusion;

    Headlight headlight{};
    headlight.position = params.camEye;
    headlight.direction = params.camW;
    headlight.cutoff = glm::cos(glm::radians(12.5f));
    headlight.outerCutoff = glm::cos(glm::radians(17.5f));

    const glm::vec3 N = glm::normalize(normal);
    const glm::vec3 V = glm::normalize(params.camEye - hitPos);
    const glm::vec3 L = glm::normalize(params.lightPos - hitPos);
    const glm::vec3 H = glm::normalize(V + L);

    // BRDF implementation of Frostbite.
    glm::vec3 brdf(0.0f);

    const float theta = glm::dot(L, V);
    //if(theta > headlight.outerCutoff) {
        // Lambertian diffuse term (diffuse reflectance = sigma).
        const glm::vec3 diffuseColor = (1.0f - metallic) * glm::vec3(baseColor) * glm::max(0.f, glm::dot(L, N));
        const glm::vec3 fd = diffuseColor /*/ glm::pi<float>()*/;

        glm::vec3 fr { 0.f, 0.f, 0.f };
        // Specular term.
        const float dotNV = glm::dot(N, V);
        const float dotNL = glm::dot(N, L);
        //const float dotNV = glm::abs(glm::dot(N, V)) + glm::epsilon<float>();
        if (dotNV > 0 && dotNL > 0) {
            const float dotNH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f);
            const float dotLH = glm::clamp(glm::dot(L, H), 0.0f, 1.0f);

            const float alpha = roughness * roughness;
            const glm::vec3 f0 = glm::mix(glm::vec3(0.5f * alpha), glm::vec3(baseColor), metallic);

            const float D = dGgx(dotNH, alpha);
            const float G = gGgx(dotNL, dotNV, alpha);
            const glm::vec3 F = fSchlick(f0, dotLH);

            fr = glm::abs( (D * G * F) / glm::pi<float>());
        }

        brdf = (fd + fr);// * headlight.intensity(theta) * headlight.attenuation(glm::length(params.lightPos - hitPos));
        //brdf = glm::vec3(dotNL);
    //}

    setPayload(ambient + brdf + emissive);
    //setPayload(brdf);
    //setPayload(n);
}