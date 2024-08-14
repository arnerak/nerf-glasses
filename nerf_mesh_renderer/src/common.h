#pragma once

// == Includes =========================================================================================================

#include <spdlog/spdlog.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/epsilon.hpp>


#include <cuda_fp16.h>
#include <vector_types.h>

#include <cstdint>
#include <string>
#include <stdexcept>

// == Macros ===========================================================================================================

#define EPS 0.0001f

#define NMR_PI 3.14159265358979323846f

#define HOST_DEVICE __host__ __device__

#define PRAGMA_UNROLL #pragma unroll

#define NGP_NAMESPACE_BEGIN namespace ngp {
#define NGP_NAMESPACE_END }

/** To enforce a semi-colon after calling macros like GL_CHECK and have self-documenting code where needed. */
#define NOOP ((void)0)

/** Used to make explicit fallthrough cases in switch statements. (Idea from Blender style-guide.) */
#define SWITCH_FALLTHROUGH NOOP

#ifdef NDEBUG

#define GL_CHECK(glCall) (glCall)
#define CUDA_CHECK(cudaCall) (cudaCall)
#define OPTIX_CHECK(optixCall) (optixCall)

#else

#define GL_CHECK(glCall)                                    \
    {                                                       \
        glCall;                                             \
        const GLenum result = glGetError();                 \
        if(result != GL_NO_ERROR) {                         \
            spdlog::error("OpenGL error: {}:{}, code: {}",  \
                __FILE__, __LINE__, result);                \
        }                                                   \
    }                                                       \
    NOOP

#define CUDA_CHECK(cudaCall)                                                \
    {                                                                       \
        const cudaError_t result = cudaCall;                                \
        if(result != cudaSuccess) {                                         \
            spdlog::error("CUDA error: {}:{}, code: {}, reason: {}",        \
                __FILE__, __LINE__, result, cudaGetErrorString(result));    \
        }                                                                   \
    }                                                                       \
    NOOP

#define OPTIX_CHECK(optixCall)                                              \
    {                                                                       \
        const OptixResult result = optixCall;                               \
        if(result != OPTIX_SUCCESS) {                                       \
            spdlog::error("OptiX error: {}:{}, code: {}, reason: {}",       \
                __FILE__, __LINE__, result, optixGetErrorString(result));   \
        }                                                                   \
    }                                                                       \
    NOOP

#endif


// == Helper methods ===================================================================================================

namespace common {
    /** To ensure that output dimensions are at least 1 in both x and y to avoid an error with cudaMalloc. */
    void ensureMinimumSize(uint32_t& width, uint32_t& height);
}

