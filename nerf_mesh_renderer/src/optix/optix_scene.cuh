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

#include <optix.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <Eigen/Core>

#include <cstdint>

struct Params {
    uchar4* image;
    uint32_t imageWidth;
    uint32_t imageHeight;
    glm::vec3 camEye;
    glm::vec3 camU;
    glm::vec3 camV;
    glm::vec3 camW;
    glm::vec3 lightPos;
    OptixTraversableHandle handle;
    bool bNoNerfsRendered;
    float* dDepthBuffer;
    Eigen::Array4f* dFramebuffer;
};

struct GeometryData {
    struct Mesh {
        uint16_t* dIndices;
        glm::vec3* dNormals;
        glm::vec4* dTangents;
        glm::vec2* dTexCoords;

        glm::vec3 emissiveFactor;
        cudaTextureObject_t emissiveTexture;

        glm::vec4 baseColorFactor;
        cudaTextureObject_t baseColorTexture;

        float metallicFactor;
        float roughnessFactor;
        cudaTextureObject_t metallicRoughnessTexture;

        float normalScale;
        cudaTextureObject_t normalTexture;

        float occlusionStrength;
        cudaTextureObject_t occlusionTexture;
    };

    union {
        Mesh mesh;
    };
};

struct RayGenData {};

struct MissData {
    float3 backgroundColor;
};

struct HitGroupData {
    GeometryData geometryData;
};