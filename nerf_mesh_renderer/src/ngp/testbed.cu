/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include "testbed.cuh"

#include <glad/gl.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <fstream>
#include <set>

#include "json_binding.h"
#include "nerf_loader.cuh"
#include "nerf_network.cuh"
#include "render_buffer.cuh"
#include "random_val.cuh"
#include "ngp_common.cuh"

#include <iostream>

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far


using namespace Eigen;
using namespace std::literals::chrono_literals;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

Testbed::Testbed(std::string name) : m_name{std::move(name)} {
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
        throw std::runtime_error{"Testbed requires CUDA 10.2 or later."};
    }

    uint32_t compute_capability = cuda_compute_capability();
    if (compute_capability < MIN_GPU_ARCH) {
//        tlog::warning() << "Insufficient compute capability " << compute_capability << " detected.";
//        tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
    }

    m_network_config = {
            {"loss", {
                             {"otype", "L2"}
                     }},
            {"optimizer", {
                             {"otype", "Adam"},
                             {"learning_rate", 1e-3},
                             {"beta1", 0.9f},
                             {"beta2", 0.99f},
                             {"epsilon", 1e-15f},
                             {"l2_reg", 1e-6f},
                     }},
            {"encoding", {
                             {"otype", "HashGrid"},
                             {"n_levels", 16},
                             {"n_features_per_level", 2},
                             {"log2_hashmap_size", 19},
                             {"base_resolution", 16},
                     }},
            {"network", {
                             {"otype", "FullyFusedMLP"},
                             {"n_neurons", 64},
                             {"n_layers", 2},
                             {"activation", "ReLU"},
                             {"output_activation", "None"},
                     }},
    };

    reset_camera();

    set_exposure(0);
    set_min_level(0.f);
    set_max_level(1.f);
}


// == HOST_DEVICE & kernels ============================================================================================

// =============
// == Loading ==
// =============

inline constexpr __device__ uint32_t NERF_CASCADES() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

inline HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip) {
    return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
                                  const uint8_t* __restrict__ prev_level,
                                  uint8_t* __restrict__ next_level
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    uint8_t bits = 0;

    PRAGMA_UNROLL
    for (uint8_t j = 0; j < 8; ++j) {
        // If any bit is set in the previous level, set this
        // level's bit. (Max pooling.)
        bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
    }

    uint32_t x = tcnn::morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
    uint32_t y = tcnn::morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
    uint32_t z = tcnn::morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

    next_level[tcnn::morton3D(x, y, z)] |= bits;
}

__global__ void grid_to_bitfield(
        const uint32_t n_elements,
        const uint32_t n_nonzero_elements,
        const float* __restrict__ grid,
        uint8_t* __restrict__ grid_bitfield,
        const float* __restrict__ mean_density_ptr
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    if (i >= n_nonzero_elements) {
        grid_bitfield[i] = 0;
        return;
    }

    uint8_t bits = 0;

    float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

    PRAGMA_UNROLL
    for (uint8_t j = 0; j < 8; ++j) {
        bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
    }

    grid_bitfield[i] = bits;
}

// ===============
// == Rendering ==
// ===============

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }

inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length

inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch

inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }

// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

inline __device__ int mip_from_pos(const Vector3f& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
    int exponent;
    float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
    frexpf(maxval, &exponent);
    return min(max_cascade, max(0, exponent+1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
    int mip = mip_from_pos(pos, max_cascade);
    dt *= 2*NERF_GRIDSIZE();
    if (dt<1.f) return mip;
    int exponent;
    frexpf(dt, &exponent);
    return min(max_cascade, max(exponent, mip));
}

__device__ Vector3f warp_position(const Vector3f& pos, const BoundingBox& aabb) {
    // return {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f), tcnn::logistic(pos.z() - 0.5f)};
    // return pos;
    return aabb.relative_pos(pos);
}

__device__ Vector3f unwarp_position(const Vector3f& pos, const BoundingBox& aabb) {
    // return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
    // return pos;
    return aabb.min + pos.cwiseProduct(aabb.diag());
}

HOST_DEVICE Vector3f warp_direction(const Vector3f& dir) {
    return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ float warp_dt(float dt) {
    float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
    return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ float unwarp_dt(float dt) {
    float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
    return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

inline HOST_DEVICE float calc_dt(float t, float cone_angle) {
    return tcnn::clamp(t*cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());
}

HOST_DEVICE uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip) {
    float mip_scale = scalbnf(1.0f, -mip);
    pos -= Vector3f::Constant(0.5f);
    pos *= mip_scale;
    pos += Vector3f::Constant(0.5f);

    Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

    if (i.x() < -1 || i.x() > NERF_GRIDSIZE() || i.y() < -1 || i.y() > NERF_GRIDSIZE() || i.z() < -1 || i.z() > NERF_GRIDSIZE()) {
        printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
    }

    uint32_t idx = tcnn::morton3D(
            tcnn::clamp(i.x(), 0, (int)NERF_GRIDSIZE()-1),
            tcnn::clamp(i.y(), 0, (int)NERF_GRIDSIZE()-1),
            tcnn::clamp(i.z(), 0, (int)NERF_GRIDSIZE()-1)
    );

    return idx;
}


uint32_t Testbed::Nerf::pos_to_cascaded_grid_idx(Vector3f pos, uint32_t mip) {
    
    return cascaded_grid_idx_at(pos, mip);
}

__device__ bool density_grid_occupied_at(const Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
    uint32_t idx = cascaded_grid_idx_at(pos, mip);
    return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}

// expects a NERF_GRIDSIZExNERF_GRIDSIZExNERF_GRIDSIZE grid
// __global__ void density_grid_to_regular_grid(int* grid) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     const int idz = blockIdx.z * blockDim.z + threadIdx.z;
//     uint32_t idx = cascaded_grid_idx_at(pos, 0);
// }

__global__ void density_grid_from_regular_grid(int* grid, const uint8_t* density_grid_bitfield, uint8_t* new_density_grid_bitfield) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;

    Vector3f pos = Vector3f{ idx, idy, idz } / (float)NERF_GRIDSIZE();

    uint32_t mip = 0;

    uint32_t idx_bitfield = cascaded_grid_idx_at(pos, mip);
    uint32_t idx_grid = idz * NERF_GRIDSIZE() * NERF_GRIDSIZE() + idy * NERF_GRIDSIZE() + idx;

    uint8_t density_byte = density_grid_bitfield[idx_bitfield/8+grid_mip_offset(mip)/8];

    // set nth bit to what's stored in regular grid
    density_byte = (density_byte & ~(1UL << (idx_bitfield%8))) | (grid[idx_grid]<<(idx_bitfield%8));
}


inline __device__ float distance_to_next_voxel(const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) { // dda like step
    Vector3f p = res * pos;
    float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
    float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
    float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
    float t = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) {
    // Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
    // due to the different stepping.
    // float dt = calc_dt(t, cone_angle);
    // return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
    do {
        t += calc_dt(t, cone_angle);
    } while (t < t_target);
    return t;
}

inline HOST_DEVICE float calc_cone_angle(float cosine, const Eigen::Vector2f& focal_length, float cone_angle_constant) {
    // Pixel size. Doesn't always yield a good performance vs. quality
    // trade off. Especially if training pixels have a much different
    // size than rendering pixels.
    // return cosine*cosine / focal_length.mean();
    return cone_angle_constant;
}

__device__ float network_to_density(float val, ENerfActivation activation) {
    switch (activation) {
        case ENerfActivation::None: return val;
        case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
        case ENerfActivation::Logistic: return tcnn::logistic(val);
        case ENerfActivation::Exponential: return __expf(val);
        default: assert(false);
    }
    return 0.0f;
}

__device__ float network_to_rgb(float val, ENerfActivation activation) {
    switch (activation) {
        case ENerfActivation::None: return val;
        case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
        case ENerfActivation::Logistic: return tcnn::logistic(val);
        case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
        default: assert(false);
    }
    return 0.0f;
}

__device__ Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation) {
    return {
            network_to_rgb(float(local_network_output[0]), activation),
            network_to_rgb(float(local_network_output[1]), activation),
            network_to_rgb(float(local_network_output[2]), activation)
    };
}

__global__ void init_rays_with_payload_kernel_nerf(
        uint32_t sample_index,
        NerfPayload* __restrict__ payloads,
        Vector2i resolution,
        Vector2f focal_length,
        Matrix<float, 3, 4> camera_matrix0,
        Matrix<float, 3, 4> camera_matrix1,
        Vector4f rolling_shutter,
        Vector2f screen_center,
        Vector3f parallax_shift,
        bool snap_to_pixel_centers,
        BoundingBox render_aabb,
        Matrix4f render_aabb_to_local,
        float near_distance,
        float plane_z,
        float aperture_size,
        Lens lens,
        const float* __restrict__ envmap_data,
        const Vector2i envmap_resolution,
        Array4f* __restrict__ framebuffer,
        float* __restrict__ depthbuffer,
        const float* __restrict__ distortion_data,
        const Vector2i distortion_resolution,
        Vector2i quilting_dims
) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= resolution.x() || y >= resolution.y()) {
        return;
    }

    uint32_t idx = x + resolution.x() * y;

    if (plane_z < 0) {
        aperture_size = 0.0;
    }

    if (quilting_dims != Vector2i::Ones()) {
        apply_quilting(&x, &y, resolution, parallax_shift, quilting_dims);
    }

    // TODO: pixel_to_ray also immediately computes u,v for the pixel, so this is somewhat redundant
    float u = (x + 0.5f) * (1.f / resolution.x());
    float v = (y + 0.5f) * (1.f / resolution.y());
    float ray_time = rolling_shutter.x() + rolling_shutter.y() * u + rolling_shutter.z() * v + rolling_shutter.w() * ld_random_val(sample_index, idx * 72239731);
    Ray ray = pixel_to_ray(
            sample_index,
            {x, y},
            resolution.cwiseQuotient(quilting_dims),
            focal_length,
            camera_matrix0 * ray_time + camera_matrix1 * (1.f - ray_time),
            screen_center,
            parallax_shift,
            snap_to_pixel_centers,
            near_distance,
            plane_z,
            aperture_size,
            lens,
            distortion_data,
            distortion_resolution
    );

    NerfPayload& payload = payloads[idx];
    payload.max_weight = 0.0f;

    if (plane_z < 0) {
        float n = ray.d.norm();
        payload.origin = ray.o;
        payload.dir = (1.0f/n) * ray.d;
        payload.t = -plane_z*n;
        payload.idx = idx;
        payload.n_steps = 0;
        payload.alive = false;
        depthbuffer[idx] = -plane_z;
        return;
    }

    depthbuffer[idx] = 1e10f;

    ray.d = ray.d.normalized();

    if (envmap_data) {
        // TODO kebiro: read_envmap() from envmap.cuh ... needed?
//        framebuffer[idx] = read_envmap(envmap_data, envmap_resolution, ray.d);
    }

    auto transformed_dir = render_aabb_to_local.block<3,3>(0,0) * ray.d;
    Eigen::Vector3f translation = render_aabb_to_local.block<3,1>(0,3);
    render_aabb_to_local.block<3,1>(0,3) = Eigen::Vector3f{ 0.5f, 0.5f, 0.5f };
    Eigen::Vector3f transformed_origin = (render_aabb_to_local * ray.o.homogeneous()).head<3>();
    transformed_origin += render_aabb_to_local.block<3,3>(0,0) * translation;
    
    float t = fmaxf(render_aabb.ray_intersect(transformed_origin, transformed_dir).x(), 0.0f) + 1e-6f;

    //if (!render_aabb.contains(transformed_origin + transformed_dir * t)) {
    //    payload.origin = ray.o;
    //    payload.dir = ray.d;
    //    payload.idx = idx;
    //    payload.alive = false;
    //    return;
    //}

    payload.origin = transformed_origin;
    payload.dir = transformed_dir;
    payload.t = t;
    payload.t_start = 0;
    payload.idx = idx;
    payload.n_steps = 0;
    payload.alive = render_aabb.contains(transformed_origin + transformed_dir * t);
    //payload.t_surface = 0.f;
    //payload.surface_color = { 0.f, 0.f, 0.f, 0.f };
}

// ray initialization routine
__global__ void advance_pos_nerf(
        const uint32_t n_elements,
        BoundingBox render_aabb,
        Matrix3f render_aabb_to_local,
        Vector3f camera_fwd,
        Vector2f focal_length,
        uint32_t sample_index,
        NerfPayload* __restrict__ payloads,
        const uint8_t* __restrict__ density_grid,
        uint32_t min_mip,
        float cone_angle_constant
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive) {
        if (payload.t_surface) {
            payload.t = payload.t_surface;
            payload.alive = true;
        }
        return;
    }

    Vector3f origin = payload.origin;
    Vector3f dir = payload.dir;
    Vector3f idir = dir.cwiseInverse();

    float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

    float t = payload.t;
    float dt = calc_dt(t, cone_angle);
    t += ld_random_val(sample_index, i * 786433) * dt;
    Vector3f pos;

    while (1) {
        // we've marched beyond a mesh surface, stop marching
        if (payload.t_surface && t > payload.t_surface) {
            payload.t = payload.t_surface;
            return;
        }

        pos = origin + dir * t;
        if (!render_aabb.contains(render_aabb_to_local * pos)) {
            if (payload.t_surface) {
                payload.t = payload.t_surface;
                return;
            }
            payload.alive = false;
            break;
        }

        dt = calc_dt(t, cone_angle);
        uint32_t mip = max(min_mip, mip_from_dt(dt, pos));

        if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
            break;
        }

        uint32_t res = NERF_GRIDSIZE()>>mip;
        t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
    }

    payload.t = t;
    if (mip_from_pos(origin + dir * t) == 0)
        payload.t_start = t;
}

__global__ void compact_kernel_nerf(
        const uint32_t n_elements,
        Array4f* src_rgba, float* src_depth, NerfPayload* src_payloads,
        Array4f* dst_rgba, float* dst_depth, NerfPayload* dst_payloads,
        Array4f* dst_final_rgba, float* dst_final_depth, NerfPayload* dst_final_payloads,
        uint32_t* counter, uint32_t* finalCounter
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    NerfPayload& src_payload = src_payloads[i];

    if (src_payload.alive) {
        uint32_t idx = atomicAdd(counter, 1);
        dst_payloads[idx] = src_payload;
        dst_rgba[idx] = src_rgba[i];
        dst_depth[idx] = src_depth[i];
    } else if (src_rgba[i].w() > 0.001f) {
        uint32_t idx = atomicAdd(finalCounter, 1);
        dst_final_payloads[idx] = src_payload;
        dst_final_rgba[idx] = src_rgba[i];
        dst_final_depth[idx] = src_depth[i];
    }
}

__global__ void generate_next_nerf_network_inputs(
        const uint32_t n_elements,
        BoundingBox render_aabb,
        Matrix3f render_aabb_to_local,
        BoundingBox train_aabb,
        Vector2f focal_length,
        Vector3f camera_fwd,
        NerfPayload* __restrict__ payloads,
        PitchedPtr<NerfCoordinate> network_input,
        uint32_t n_steps,
        const uint8_t* __restrict__ density_grid,
        uint32_t min_mip,
        float cone_angle_constant,
        const float* extra_dims
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive) {
        return;
    }

    Vector3f origin = payload.origin;
    Vector3f dir = payload.dir;
    Vector3f idir = dir.cwiseInverse();

    float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

    float t = payload.t;

    for (uint32_t j = 0; j < n_steps; ++j) {
        Vector3f pos;
        float dt = 0.0f;

        while (1) {
            // we've marched beyond a mesh surface, stop marching
            if (payload.t_surface && t > payload.t_surface && payload.surface_color.w() == 1.f) {

                payload.n_steps = j;
                payload.t = payload.t_surface;
                return;
            }

            pos = origin + dir * t;
            if (!render_aabb.contains(render_aabb_to_local * pos)) {
                payload.n_steps = j;
                return;
            }

            dt = calc_dt(t - payload.t_start, cone_angle);
            //dt = MIN_CONE_STEPSIZE();
            uint32_t mip = max(min_mip, mip_from_dt(dt, pos));

            if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
                break;
            }

            uint32_t res = NERF_GRIDSIZE()>>mip;
            t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
        }

        network_input(i + j * n_elements)->set_with_optional_extra_dims(warp_position(pos, train_aabb), warp_direction(dir), warp_dt(dt), extra_dims, network_input.stride_in_bytes); // XXXCONE
        t += dt;
    }

    payload.t = t;
    payload.n_steps = n_steps;
}


__device__ bool mesh_grid_occupied_at(const Vector3f& pos) {
	
	return true;
}

__device__ bool sphere_intersect(const Ray &ray, const Vector3f& center, float r, Vector3f& intersection, Vector3f& normal, float& t) {
	Vector3f p = ray.o - center;
	float r_sq = r * r;
	float p_d = p.dot(ray.d);
	if (p_d > 0 || p.dot(p) < r_sq)
		return false;
	Vector3f a = p - p_d * ray.d;
	float a_sq = a.dot(a);
	if (a_sq > r_sq)
		return false;
	float h = sqrtf(r_sq - a_sq);
	Vector3f i = a - h * ray.d;
	intersection = center + i;
	normal = i / r;
	t = (ray.o - intersection).norm();
	return true;
} 

__global__ void transform_rays(
    const uint32_t n_elements, 
    NerfPayload* __restrict__ payloads,
    Eigen::Matrix4f transform
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    auto& payload = payloads[i];

    //Eigen::Vector3f origin_new = { payload.origin.x(), payload.origin.y() + 0.4f, payload.origin.z() };

    payload.origin = (transform * payload.origin.homogeneous()).head<3>();
    payload.dir = transform.block<3,3>(0,0) * payload.dir;

    //if (payload.t_surface) {
    //    Eigen::Vector3f intersection_point = payload.origin + payload.dir * payload.t_surface;
    //    Eigen::Vector3f asd = intersection_point - origin_new;
    //    //payload.t_surface = sqrtf()
    //}

    //payloads[i].origin.x() += 0.01f;
}

__global__ void generate_intersection(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	Matrix3f render_aabb_to_local,
	Vector2f focal_length,
	Vector3f camera_fwd,
	NerfPayload* __restrict__ payloads,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	//if (!payload.alive) {
	//	return;
	//}

	Vector3f origin = payload.origin;
	Vector3f dir = payload.dir;
	Vector3f idir = dir.cwiseInverse();

	Vector3f light_dir{ Vector3f{1.f, 0.f, 1.f }.normalized() };

	float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

	float t = payload.t;

	Vector3f normal, intersection;
	if (mesh_grid_occupied_at(origin) && 
		sphere_intersect({ origin, dir }, {0.5f, 0.5f, 0.5f}, 0.3f, intersection, normal, t)) {
		payload.surface_color.head<3>() = Vector3f{0.6f, 0.f, 0.f} * max(normal.dot(light_dir), 0.f) + Vector3f{0.4f, 0.0f, 0.0f};
		payload.surface_color.w() = 1.f;
		payload.t_surface = t;
	}
}

__global__ void check_collision(
        const uint32_t n_elements,
        const uint32_t stride,
        const uint32_t current_step,
        BoundingBox aabb,
        NerfPayload* payloads,
        PitchedPtr<NerfCoordinate> network_input,
        const tcnn::network_precision_t* __restrict__ network_output,
        uint32_t padded_output_width,
        uint32_t n_steps,
        ENerfActivation density_activation,
        float* collision_distances,
        uint32_t* counter
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive) {
        return;
    }

    Vector3f origin = payload.origin;
    // Composite in the last n steps
    uint32_t actual_n_steps = payload.n_steps;
    uint32_t j = 0;

    for (; j < actual_n_steps; ++j) {
        tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output;
        local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
        local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
        local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
        local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
        const NerfCoordinate* input = network_input(i + j * n_elements);
        Vector3f warped_pos = input->pos.p;
        Vector3f pos = unwarp_position(warped_pos, aabb);


        //float T = 1.f - local_rgba.w();
        float dt = unwarp_dt(input->dt);

        float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
        //float weight = alpha * T;

        if (alpha > 0.f) {
            payload.alive = false;
            collision_distances[i] = (pos - payload.origin).norm();
            return;
        }
    }

    atomicAdd(counter, 1);

    // if (j < n_steps) {

    //     payload.alive = false;
    //     payload.n_steps = j + current_step;
    // }


}

__global__ void composite_kernel_nerf(
        const uint32_t n_elements,
        const uint32_t stride,
        const uint32_t current_step,
        BoundingBox aabb,
        float glow_y_cutoff,
        int glow_mode,
        const uint32_t n_training_images,
        const TrainingXForm* __restrict__ training_xforms,
        Matrix<float, 3, 4> camera_matrix,
        Vector2f focal_length,
        float depth_scale,
        Array4f* __restrict__ rgba,
        float* __restrict__ depth,
        NerfPayload* payloads,
        PitchedPtr<NerfCoordinate> network_input,
        const tcnn::network_precision_t* __restrict__ network_output,
        uint32_t padded_output_width,
        uint32_t n_steps,
        const uint8_t* __restrict__ density_grid,
        ENerfActivation rgb_activation,
        ENerfActivation density_activation,
        int show_accel,
        float min_transmittance
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive) {
        //if (payload.t_surface) {
        //    rgba[i] = payload.surface_color;
        //    depth[i] = payload.t_surface;
        //}
        return;
    }

    Array4f local_rgba = rgba[i];
    float local_depth = depth[i];
    Vector3f origin = payload.origin;
    Vector3f cam_fwd = camera_matrix.col(2);
    // Composite in the last n steps
    uint32_t actual_n_steps = payload.n_steps;
    uint32_t j = 0;

    for (; j < actual_n_steps; ++j) {
        tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output;
        local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
        local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
        local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
        local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
        const NerfCoordinate* input = network_input(i + j * n_elements);
        Vector3f warped_pos = input->pos.p;
        Vector3f pos = unwarp_position(warped_pos, aabb);


        float T = 1.f - local_rgba.w();
        float dt = unwarp_dt(input->dt);
        if (payload.t > payload.t_surface && payload.surface_color.w() > 0) {
            local_rgba.head<3>() += Array3f { payload.surface_color.head<3>() } * payload.surface_color.w() * T;
            local_rgba.w() += payload.surface_color.w() * T;

            //if (payload.surface_color.w() < 0.95) {
            //    local_rgba.head<3>() = Array3f { 1.f, 0.f, 0.f };
            //}
            payload.surface_color.w() = 0.f;
            T = 1.f - local_rgba.w();
            if (local_rgba.w() > 0.99f) {
               local_rgba /= local_rgba.w();
               //local_rgba = Array4f{ 0.f, 0.f, 1.f, 1.f };
               break;
            }
        }

        float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
        if (show_accel >= 0) {
            alpha = 1.f;
        }
        float weight = alpha * T;

        Array3f rgb = network_to_rgb(local_network_output, rgb_activation);


        if (weight > local_rgba.w()) {
            //local_depth = payload.t;
        }
        
        local_rgba.head<3>() += rgb * weight;
        local_rgba.w() += weight;
        if (weight > payload.max_weight) {
            payload.max_weight = weight;
            local_depth = (pos - camera_matrix.col(3)).norm();
            //local_depth = cam_fwd.dot(pos - camera_matrix.col(3));
        }

        if (local_rgba.w() > (1.0f - min_transmittance)) {
            local_rgba /= local_rgba.w();
            break;
        }
    }

    if (j < n_steps) {
		if (payload.surface_color.w() > 0) {
			local_rgba += Array4f{ payload.surface_color } * (1.f - local_rgba.w());
			//float depth = payload.t * 0.25f;
			//local_rgba = Array4f{ depth, depth, depth, 1.f };

            // if (payload.surface_color.w() < 0.95) {
            //     local_rgba = Array4f{ 1.f * payload.surface_color.w(), 0.f, 0.f, payload.surface_color.w() };
            //     //float depth = payload.t * 0.25f;
            //     //local_rgba = Array4f{ depth, depth, depth, 1.f };
            // }
		}

        payload.alive = false;
        payload.n_steps = j + current_step;
    }

    rgba[i] = local_rgba;
    depth[i] = local_depth;
}

__global__ void shade_kernel_nerf(
        const uint32_t n_elements,
        Array4f* __restrict__ rgba,
        float* __restrict__ depth,
        NerfPayload* __restrict__ payloads,
        bool train_in_linear_colors,
        Array4f* __restrict__ frame_buffer,
        float* __restrict__ depth_buffer
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    NerfPayload& payload = payloads[i];

    Array4f tmp = rgba[i];

    if (!train_in_linear_colors) {
        // Accumulate in linear colors
        tmp.head<3>() = srgb_to_linear(tmp.head<3>());
    }

    frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.w());
    if (tmp.w() > 0.2f) {
        depth_buffer[payload.idx] = depth[i];
    }
}

// == Testbed ==========================================================================================================

// =============
// == Loading ==
// =============

void Testbed::load_snapshot(const std::string& filepath_string) {
    auto config = load_network_config(filepath_string);
    if (!config.contains("snapshot")) {
        throw std::runtime_error{fmt::format("File {} does not contain a snapshot.", filepath_string)};
    }

    const auto& snapshot = config["snapshot"];

    if (snapshot.value("version", 0) < 1) {
        throw std::runtime_error{"Snapshot uses an old format."};
    }

    m_aabb = snapshot.value("aabb", m_aabb);
    m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

    if (snapshot["density_grid_size"] != NERF_GRIDSIZE()) {
        throw std::runtime_error{"Incompatible grid size."};
    }

    m_nerf.training.counters_rgb.rays_per_batch = snapshot["nerf"]["rgb"]["rays_per_batch"];
    m_nerf.training.counters_rgb.measured_batch_size = snapshot["nerf"]["rgb"]["measured_batch_size"];
    m_nerf.training.counters_rgb.measured_batch_size_before_compaction = snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"];

    // If we haven't got a nerf dataset loaded, load dataset metadata from the snapshot
    // and render using just that.
    if (m_data_path.empty() && snapshot["nerf"].contains("dataset")) {
        m_nerf.training.dataset = snapshot["nerf"]["dataset"];
//        load_nerf();
    } else {
        if (snapshot["nerf"].contains("aabb_scale")) {
            m_nerf.training.dataset.aabb_scale = snapshot["nerf"]["aabb_scale"];
        }
    }

    load_nerf_post();

    GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
    m_nerf.density_grid.resize(density_grid_fp16.size());

    parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
        density_grid[i] = (float)density_grid_fp16[i];
    });

    if (m_nerf.density_grid.size() == NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * (m_nerf.max_cascade + 1)) {
        update_density_grid_mean_and_bitfield(nullptr);
    } else if (m_nerf.density_grid.size() != 0) {
        // A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
        throw std::runtime_error{"Incompatible number of grid cascades."};
    }

    // Needs to happen after `load_nerf_post()`
    if (snapshot.contains("render_aabb_to_local")) from_json(snapshot.at("render_aabb_to_local"), m_render_aabb_to_local);
    m_render_aabb = snapshot.value("render_aabb", m_render_aabb);

    m_network_config_path = filepath_string;
    m_network_config = config;

    reset_network(false);

    m_training_step = m_network_config["snapshot"]["training_step"];
    m_loss_scalar.set(m_network_config["snapshot"]["loss"]);

    m_trainer->deserialize(m_network_config["snapshot"]);
}

json Testbed::load_network_config(const fs::path& network_config_path) {
    if (!network_config_path.empty()) {
        m_network_config_path = network_config_path;
    }

//    spdlog::debug("Loading network config from: {}", network_config_path.str());

    if (network_config_path.empty() || !network_config_path.exists()) {
        throw std::runtime_error{fmt::format("Network config {} does not exist.", network_config_path.str())};
    }

    json result;
    if (equals_case_insensitive(network_config_path.extension(), "json")) {
        throw std::runtime_error("kebiro: only supporting .msgpack");
    } else if (equals_case_insensitive(network_config_path.extension(), "msgpack")) {
        std::ifstream f{network_config_path.str(), std::ios::in | std::ios::binary};
        result = json::from_msgpack(f);
        // we assume parent pointers are already resolved in snapshots.
    }

    return result;
}

void Testbed::load_nerf_post() { // moved the second half of load_nerf here
    m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

    m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

    m_nerf.training.dataset.update_metadata();

    m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, Vector3f::Zero());
    m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

    m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Array3f>(1e-3f));
    m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Vector3f>(1e-4f));
    m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
    m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-5f);

    m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, Vector3f::Zero());
    m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

    m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, Array3f::Zero());
    m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
    m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

    m_nerf.training.cam_focal_length_gradient = Vector2f::Zero();
    m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

    m_nerf.training.reset_extra_dims(m_rng);

    if (m_nerf.training.dataset.has_rays) {
        m_nerf.training.near_distance = 0.0f;
        // m_nerf.training.optimize_exposure = true;
    }

    // Uncomment the following line to see how the network learns distortion from scratch rather than
    // starting from the distortion that's described by the training data.
    // m_nerf.training.dataset.camera = {};

    // Perturbation of the training cameras -- for debugging the online extrinsics learning code
    float perturb_amount = 0.0f;
    if (perturb_amount > 0.f) {
        for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
            Vector3f rot = random_val_3d(m_rng) * perturb_amount;
            float angle = rot.norm();
            rot /= angle;
            auto trans = random_val_3d(m_rng);
            m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0);
            m_nerf.training.dataset.xforms[i].start.col(3) += trans * perturb_amount;
            m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0);
            m_nerf.training.dataset.xforms[i].end.col(3) += trans * perturb_amount;
        }
    }

    m_nerf.training.update_transforms();

    if (!m_nerf.training.dataset.metadata.empty()) {
        m_nerf.render_lens = m_nerf.training.dataset.metadata[0].lens;
        m_screen_center = Eigen::Vector2f::Constant(1.f) - m_nerf.training.dataset.metadata[0].principal_point;
    }

    if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
        throw std::runtime_error{fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.", m_nerf.training.dataset.aabb_scale)};
    }

    int max_aabb_scale = 1 << (NERF_CASCADES()-1);
    if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
        throw std::runtime_error{fmt::format(
                "NeRF dataset must have `aabb_scale <= {}`, but is {}. "
                "You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
                max_aabb_scale, m_nerf.training.dataset.aabb_scale
        )};
    }

    m_aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
    m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
    m_raw_aabb = m_aabb;
    m_render_aabb = m_aabb;
    m_render_aabb_to_local = m_nerf.training.dataset.render_aabb_to_local;
    if (!m_nerf.training.dataset.render_aabb.is_empty()) {
        m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
    }

    m_nerf.max_cascade = 0;
    while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
        ++m_nerf.max_cascade;
    }
    std::cout << "aabb_scale: " << m_nerf.training.dataset.aabb_scale << std::endl; // From kebiro

    // Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
    // stepping in larger scenes.
    m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

    m_up_dir = m_nerf.training.dataset.up;
}

void Testbed::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
    const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();

    size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
    m_nerf.density_grid_bitfield.enlarge(size_including_mips);
    m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

    CUDA_CHECK(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
    reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

    linear_kernel(grid_to_bitfield, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

    for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
        linear_kernel(bitfield_max_pool, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
    }
}

void Testbed::reset_network(bool clear_density_grid) {
    m_rng = default_rng_t{m_seed};

    // Start with a low rendering resolution and gradually ramp up
    m_render_ms.set(10000);

    reset_accumulation();
    m_nerf.training.counters_rgb.rays_per_batch = 1 << 12;
    m_nerf.training.counters_rgb.measured_batch_size_before_compaction = 0;
    m_nerf.training.n_steps_since_cam_update = 0;
    m_nerf.training.n_steps_since_error_map_update = 0;
    m_nerf.training.n_rays_since_error_map_update = 0;
    m_nerf.training.n_steps_between_error_map_updates = 128;
    m_nerf.training.error_map.is_cdf_valid = false;
    m_nerf.training.density_grid_rng = default_rng_t{m_rng.next_uint()};

    m_nerf.training.reset_camera_extrinsics();

    m_loss_graph_samples = 0;

    // Default config
    json config = m_network_config;

    json& encoding_config = config["encoding"];
    json& loss_config = config["loss"];
    json& optimizer_config = config["optimizer"];
    json& network_config = config["network"];

    auto dims = network_dims();

    m_nerf.training.loss_type = string_to_loss_type(loss_config.value("otype", "L2"));

    // Some of the Nerf-supported losses are not supported by tcnn::Loss,
    // so just create a dummy L2 loss there. The NeRF code path will bypass
    // the tcnn::Loss in any case.
    loss_config["otype"] = "L2";

    // Automatically determine certain parameters if we're dealing with the (hash)grid encoding
    if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
        encoding_config["n_pos_dims"] = dims.n_pos;

        const uint32_t n_features_per_level = encoding_config.value("n_features_per_level", 2u);

        if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
            m_num_levels = (uint32_t)encoding_config["n_features"] / n_features_per_level;
        } else {
            m_num_levels = encoding_config.value("n_levels", 16u);
        }

        m_level_stats.resize(m_num_levels);
        m_first_layer_column_stats.resize(m_num_levels);

        const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

        m_base_grid_resolution = encoding_config.value("base_resolution", 0);
        if (!m_base_grid_resolution) {
            m_base_grid_resolution = 1u << ((log2_hashmap_size) / dims.n_pos);
            encoding_config["base_resolution"] = m_base_grid_resolution;
        }

        float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube

        // Automatically determine suitable per_level_scale
        m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
        if (m_per_level_scale <= 0.0f && m_num_levels > 1) {
            m_per_level_scale = std::exp(std::log(desired_resolution * (float)m_nerf.training.dataset.aabb_scale / (float)m_base_grid_resolution) / (m_num_levels-1));
            encoding_config["per_level_scale"] = m_per_level_scale;
        }

//        spdlog::info("GridEncoding:\nNmin={}\nb={}\nF={}\n=2^{}\nL={}",
//                     m_base_grid_resolution, m_per_level_scale, n_features_per_level, log2_hashmap_size, m_num_levels);
    }

    m_loss.reset(create_loss<precision_t>(loss_config));
    m_optimizer.reset(create_optimizer<precision_t>(optimizer_config));

    size_t n_encoding_params = 0;

    m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Array3f>(1e-3f, Array3f::Zero()));
    m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Vector3f>(1e-4f, Vector3f::Zero()));
    m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
    m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-5f);

    m_nerf.training.reset_extra_dims(m_rng);

    json& dir_encoding_config = config["dir_encoding"];
    json& rgb_network_config = config["rgb_network"];

    uint32_t n_dir_dims = 3;
    uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
    m_network = m_nerf_network = std::make_shared<NerfNetwork<precision_t>>(
            dims.n_pos,
            n_dir_dims,
            n_extra_dims,
            dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
            encoding_config,
            dir_encoding_config,
            network_config,
            rgb_network_config
    );

    m_encoding = m_nerf_network->encoding();
    n_encoding_params = m_encoding->n_params() + m_nerf_network->dir_encoding()->n_params();

//    spdlog::info("Density model: {}--[{}]-->{}--[{}(neurons={}, layers={}]-->{}",
//                 dims.n_pos,
//                 std::string(encoding_config["otype"]),
//                 m_nerf_network->encoding()->padded_output_width(),
//                 std::string(network_config["otype"]),
//                 (int)network_config["n_neurons"],
//                 ((int)network_config["n_hidden_layers"]+2),
//                 1
//    );
//
//    spdlog::info("Color model: {}--[{}]-->{}+{}--[{}(neurons={}, layers={}]-->{}",
//                 n_dir_dims,
//                 std::string(dir_encoding_config["otype"]),
//                 m_nerf_network->dir_encoding()->padded_output_width(),
//                 network_config.value("n_output_dims", 16u),
//                 std::string(rgb_network_config["otype"]),
//                 (int)rgb_network_config["n_neurons"],
//                 ((int)rgb_network_config["n_hidden_layers"]+2),
//                 3
//    );

    // Create distortion map model
    {
        json& distortion_map_optimizer_config =  config.contains("distortion_map") && config["distortion_map"].contains("optimizer") ? config["distortion_map"]["optimizer"] : optimizer_config;

        m_distortion.resolution = Vector2i::Constant(32);
        if (config.contains("distortion_map") && config["distortion_map"].contains("resolution")) {
            from_json(config["distortion_map"]["resolution"], m_distortion.resolution);
        }
        m_distortion.map = std::make_shared<TrainableBuffer<2, 2, float>>(m_distortion.resolution);
        m_distortion.optimizer.reset(create_optimizer<float>(distortion_map_optimizer_config));
        m_distortion.trainer = std::make_shared<Trainer<float, float>>(m_distortion.map, m_distortion.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(loss_config)}, m_seed);
    }

    size_t n_network_params = m_network->n_params() - n_encoding_params;

//    spdlog::info("total_encoding_params={}, total_network_params={}", n_encoding_params, n_network_params);

    m_trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(m_network, m_optimizer, m_loss, m_seed);
    m_training_step = 0;
    m_training_start_time_point = std::chrono::steady_clock::now();

    // Create envmap model
    {
        json& envmap_loss_config = config.contains("envmap") && config["envmap"].contains("loss") ? config["envmap"]["loss"] : loss_config;
        json& envmap_optimizer_config =  config.contains("envmap") && config["envmap"].contains("optimizer") ? config["envmap"]["optimizer"] : optimizer_config;

        m_envmap.loss_type = string_to_loss_type(envmap_loss_config.value("otype", "L2"));

        m_envmap.resolution = m_nerf.training.dataset.envmap_resolution;
        m_envmap.envmap = std::make_shared<TrainableBuffer<4, 2, float>>(m_envmap.resolution);
        m_envmap.optimizer.reset(create_optimizer<float>(envmap_optimizer_config));
        m_envmap.trainer = std::make_shared<Trainer<float, float, float>>(m_envmap.envmap, m_envmap.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(envmap_loss_config)}, m_seed);

        if (m_nerf.training.dataset.envmap_data.data()) {
            m_envmap.trainer->set_params_full_precision(m_nerf.training.dataset.envmap_data.data(), m_nerf.training.dataset.envmap_data.size());
        }
    }

    if (clear_density_grid) {
        m_nerf.density_grid.memset(0);
        m_nerf.density_grid_bitfield.memset(0);
    }
}

void Testbed::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw) {
    if (immediate_redraw) {
        redraw_next_frame();
    }

    if (!due_to_camera_movement || !reprojection_available()) {
        m_windowless_render_surface.reset_accumulation();
        for (auto& tex : m_render_surfaces) {
            tex.reset_accumulation();
        }
    }
}

void Testbed::translate_camera(const Vector3f& rel) {
	m_camera.col(3) += m_camera.block<3,3>(0,0) * rel * m_bounding_radius;
	reset_accumulation(true);
}

void Testbed::set_nerf_camera_matrix(const Matrix<float, 3, 4>& cam) {
	m_camera = m_nerf.training.dataset.nerf_matrix_to_ngp(cam);
}

Vector3f Testbed::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void Testbed::set_look_at(const Vector3f& pos) {
	m_camera.col(3) += pos - look_at();
}

void Testbed::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera.col(3) = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Testbed::set_view_dir(const Vector3f& dir) {
	auto old_look_at = look_at();
	m_camera.col(0) = dir.cross(m_up_dir).normalized();
	m_camera.col(1) = dir.cross(m_camera.col(0)).normalized();
	m_camera.col(2) = dir.normalized();
	set_look_at(old_look_at);
}

Testbed::NetworkDims Testbed::network_dims() const {
    return network_dims_nerf();
}

Testbed::NetworkDims Testbed::network_dims_nerf() const {
    NetworkDims dims;
    dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
    dims.n_output = 4;
    dims.n_pos = sizeof(NerfPosition) / sizeof(float);
    return dims;
}

ELossType Testbed::string_to_loss_type(const std::string& str) {
    if (equals_case_insensitive(str, "L2")) {
        return ELossType::L2;
    } else if (equals_case_insensitive(str, "RelativeL2")) {
        return ELossType::RelativeL2;
    } else if (equals_case_insensitive(str, "L1")) {
        return ELossType::L1;
    } else if (equals_case_insensitive(str, "Mape")) {
        return ELossType::Mape;
    } else if (equals_case_insensitive(str, "Smape")) {
        return ELossType::Smape;
    } else if (equals_case_insensitive(str, "Huber") || equals_case_insensitive(str, "SmoothL1")) {
        // Legacy: we used to refer to the Huber loss (L2 near zero, L1 further away) as "SmoothL1".
        return ELossType::Huber;
    } else if (equals_case_insensitive(str, "LogL1")) {
        return ELossType::LogL1;
    } else {
        throw std::runtime_error{"Unknown loss type."};
    }
}

void Testbed::reset_camera() {
    m_fov_axis = 1;
    set_fov(50.625f);
    m_zoom = 1.f;
    m_screen_center = Vector2f::Constant(0.5f);
    m_scale = 1.5f;
    m_camera <<
             1.0f, 0.0f, 0.0f, 0.5f,
            0.0f, -1.0f, 0.0f, 0.5f,
            0.0f, 0.0f, -1.0f, 0.5f;
    m_camera.col(3) -= m_scale * view_dir();
    m_smoothed_camera = m_camera;
    m_up_dir = {0.0f, 1.0f, 0.0f};
    m_sun_dir = Vector3f::Ones().normalized();
    reset_accumulation();
}

void Testbed::set_fov(float val) {
    m_relative_focal_length = Vector2f::Constant(fov_to_focal_length(1, val));
}

void Testbed::set_max_level(float maxlevel) {
    if (!m_network) return;
    auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
    if (hg_enc) {
        hg_enc->set_max_level(maxlevel);
    }
    reset_accumulation();
}

void Testbed::set_min_level(float minlevel) {
    if (!m_network) return;
    auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
    if (hg_enc) {
        hg_enc->set_quantize_threshold(powf(minlevel, 4.f) * 0.2f);
    }
    reset_accumulation();
}

Eigen::Matrix<float, 3, 4> Testbed::crop_box(bool nerf_space) const {
	Eigen::Vector3f cen = m_render_aabb_to_local.transpose() * m_render_aabb.center();
	Eigen::Vector3f radius = m_render_aabb.diag() * 0.5f;
	Eigen::Vector3f x = m_render_aabb_to_local.row(0) * radius.x();
	Eigen::Vector3f y = m_render_aabb_to_local.row(1) * radius.y();
	Eigen::Vector3f z = m_render_aabb_to_local.row(2) * radius.z();
	Eigen::Matrix<float, 3, 4> rv;
	rv.col(0) = x;
	rv.col(1) = y;
	rv.col(2) = z;
	rv.col(3) = cen;
	if (nerf_space) {
		rv = m_nerf.training.dataset.ngp_matrix_to_nerf(rv, true);
	}
	return rv;
}

void Testbed::set_crop_box(Eigen::Matrix<float, 3, 4> m, bool nerf_space) {
	if (nerf_space) {
		m = m_nerf.training.dataset.nerf_matrix_to_ngp(m, true);
	}
	Eigen::Vector3f radius(m.col(0).norm(), m.col(1).norm(), m.col(2).norm());
	Eigen::Vector3f cen(m.col(3));
	m_render_aabb_to_local.row(0) = m.col(0) / radius.x();
	m_render_aabb_to_local.row(1) = m.col(1) / radius.y();
	m_render_aabb_to_local.row(2) = m.col(2) / radius.z();
	cen = m_render_aabb_to_local * cen;
	m_render_aabb.min = cen - radius;
	m_render_aabb.max = cen + radius;
}

std::vector<Eigen::Vector3f> Testbed::crop_box_corners(bool nerf_space) const {
	Eigen::Matrix<float, 3, 4> m = crop_box(nerf_space);
	std::vector<Eigen::Vector3f> rv(8);
	for (int i = 0; i < 8; ++i) {
		rv[i] = m * Eigen::Vector4f((i & 1) ? 1.f : -1.f, (i & 2) ? 1.f : -1.f, (i & 4) ? 1.f : -1.f, 1.f);
		/* debug print out corners to check math is all lined up */
		if (0) {
			std::cout << rv[i].x() << "," << rv[i].y() << "," << rv[i].z() << " [" << i << "]";
			Eigen::Vector3f mn = m_render_aabb.min;
			Eigen::Vector3f mx = m_render_aabb.max;
			Eigen::Matrix3f m = m_render_aabb_to_local.transpose();
			Eigen::Vector3f a;

			a.x() = (i&1) ? mx.x() : mn.x();
			a.y() = (i&2) ? mx.y() : mn.y();
			a.z() = (i&4) ? mx.z() : mn.z();
			a = m * a;
			if (nerf_space) {
				a = m_nerf.training.dataset.ngp_position_to_nerf(a);
			}
			std::cout << a.x() << "," << a.y() << "," << a.z() << " [" << i << "]";
		}
	}
	return rv;
}

// == Rendering =================

void Testbed::render_frame(const Matrix<float, 3, 4>& camera_matrix0, const Matrix<float, 3, 4>& camera_matrix1,
                           const Vector4f& nerf_rolling_shutter, CudaRenderBuffer& render_buffer, bool to_srgb) {
    Vector2i max_res = m_window_res.cwiseMax(render_buffer.in_resolution());

    render_buffer.clear_frame(m_stream.get());

    Vector2f focal_length = calc_focal_length(render_buffer.in_resolution(), m_fov_axis, m_zoom);
    Vector2f screen_center = render_screen_center();

    if (m_quilting_dims != Vector2i::Ones() && m_quilting_dims != Vector2i{2, 1}) {
        // In the case of a holoplay lenticular screen, m_scale represents the inverse distance of the head above the display.
        m_parallax_shift.z() = 1.0f / m_scale;
    }

    if (!m_render_ground_truth || m_ground_truth_alpha < 1.0f) {
        render_nerf(render_buffer, max_res, focal_length, camera_matrix0, camera_matrix1, nerf_rolling_shutter, screen_center, m_stream.get());
    }

    render_buffer.set_color_space(m_color_space);
    render_buffer.set_tonemap_curve(m_tonemap_curve);

    m_prev_camera = camera_matrix0;
    m_prev_scale = m_scale;

    render_buffer.accumulate(m_exposure, m_stream.get());
    render_buffer.tonemap(m_exposure, m_background_color, to_srgb ? EColorSpace::SRGB : EColorSpace::Linear, m_stream.get());

    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

Vector2f Testbed::render_screen_center() const {
    // see pixel_to_ray for how screen center is used; 0.5,0.5 is 'normal'. we flip so that it becomes the point in the original image we want to center on.
    auto screen_center = m_screen_center;
    return {(0.5f-screen_center.x())*m_zoom + 0.5f, (0.5-screen_center.y())*m_zoom + 0.5f};
}

Vector2f Testbed::calc_focal_length(const Vector2i& resolution, int fov_axis, float zoom) const {
    return m_relative_focal_length * resolution[fov_axis] * zoom;
}

void Testbed::render_nerf(CudaRenderBuffer& render_buffer, const Vector2i& max_res, const Vector2f& focal_length, const Matrix<float, 3, 4>& camera_matrix0,
                          const Matrix<float, 3, 4>& camera_matrix1, const Vector4f& rolling_shutter, const Vector2f& screen_center, cudaStream_t stream) {
    float plane_z = m_slice_plane_z + m_scale;

    const float* extra_dims_gpu = get_inference_extra_dims(stream);

    ScopeGuard tmp_memory_guard{[&]() {
        m_nerf.tracer.clear();
    }};

    // Our motion vector code can't undo f-theta and grid distortions -- so don't render these if DLSS is enabled.
    bool render_opencv_lens = m_nerf.render_with_lens_distortion && (m_nerf.render_lens.mode == ELensMode::OpenCV);
    bool render_grid_distortion = m_nerf.render_with_lens_distortion;

    Lens lens = render_opencv_lens ? m_nerf.render_lens : Lens{};

    Eigen::Matrix4f model_mat { Eigen::Matrix4f::Identity() };
    model_mat.block<3,3>(0, 0) = Eigen::Matrix3f { 
        Eigen::AngleAxisf(m_model_rotation[0] * M_PI, Vector3f::UnitX())
        * Eigen::AngleAxisf(m_model_rotation[1] * M_PI, Vector3f::UnitY())
        * Eigen::AngleAxisf(m_model_rotation[2] * M_PI, Vector3f::UnitZ()) };
    model_mat.block<3, 1>(0, 3) = Eigen::Vector3f{ m_model_translation[0], m_model_translation[1], m_model_translation[2] };

    m_nerf.tracer.init_rays_from_camera(
            render_buffer.spp(),
            m_network->padded_output_width(),
            m_nerf_network->n_extra_dims(),
            render_buffer.in_resolution(),
            focal_length,
            camera_matrix0,
            camera_matrix1,
            rolling_shutter,
            screen_center,
            m_parallax_shift,
            m_quilting_dims,
            m_snap_to_pixel_centers,
            m_render_aabb,
            m_render_aabb_to_local,
            model_mat,
            m_render_near_distance,
            plane_z,
            m_aperture_size,
            lens,
            m_envmap.envmap->params_inference(),
            m_envmap.resolution,
            render_grid_distortion ? m_distortion.map->params_inference() : nullptr,
            m_distortion.resolution,
            render_buffer.frame_buffer(),
            render_buffer.depth_buffer(),
            m_nerf.density_grid_bitfield.data(),
            m_nerf.show_accel,
            m_nerf.cone_angle_constant,
            stream
    );

    float depth_scale = 1.0f / m_nerf.training.dataset.scale;
    uint32_t n_hit = m_nerf.tracer.trace(
            *m_nerf_network,
            m_render_aabb,
            m_render_aabb_to_local,
            m_aabb,
            m_nerf.training.n_images_for_training,
            m_nerf.training.transforms.data(),
            focal_length,
            m_nerf.cone_angle_constant,
            m_nerf.density_grid_bitfield.data(),
            camera_matrix1,
            depth_scale,
            m_visualized_layer,
            m_visualized_dimension,
            m_nerf.rgb_activation,
            m_nerf.density_activation,
            m_nerf.show_accel,
            m_nerf.render_min_transmittance,
            m_nerf.glow_y_cutoff,
            m_nerf.glow_mode,
            extra_dims_gpu,
            stream
    );

    RaysNerfSoa& rays_hit = m_nerf.tracer.rays_hit();

    linear_kernel(shade_kernel_nerf, 0, stream,
                  n_hit,
                  rays_hit.rgba,
                  rays_hit.depth,
                  rays_hit.payload,
                  m_nerf.training.linear_colors,
                  render_buffer.frame_buffer(),
                  render_buffer.depth_buffer()
    );
}

const float* Testbed::get_inference_extra_dims(cudaStream_t stream) const {
    if (m_nerf_network->n_extra_dims() == 0) {
        return nullptr;
    }
    const float* extra_dims_src = m_nerf.training.extra_dims_gpu.data() + m_nerf.extra_dim_idx_for_inference * m_nerf.training.dataset.n_extra_dims();
    if (!m_nerf.training.dataset.has_light_dirs) {
        return extra_dims_src;
    }

    // the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
    // we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
    size_t size = m_nerf_network->n_extra_dims() * sizeof(float);
    float* dims_gpu = m_nerf.training.extra_dims_gpu.data() + m_nerf.training.dataset.n_images * m_nerf.training.dataset.n_extra_dims();
    CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
    Eigen::Vector3f light_dir = warp_direction(m_nerf.light_dir.normalized());
    CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(Eigen::Vector3f)), cudaMemcpyHostToDevice, stream));
    return dims_gpu;
}

// == Testbed::Nerf ====================================================================================================

uint8_t* Testbed::Nerf::get_density_grid_bitfield_mip(uint32_t mip) {
    return density_grid_bitfield.data() + grid_mip_offset(mip)/8;
}

// == Testbed::Nerf::Training ==========================================================================================

void Testbed::Nerf::Training::reset_extra_dims(default_rng_t &rng) {
    uint32_t n_extra_dims = dataset.n_extra_dims();
    std::vector<float> extra_dims_cpu(n_extra_dims * (dataset.n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
    float *dst = extra_dims_cpu.data();
    ArrayXf zero(n_extra_dims);
    zero.setZero();
    extra_dims_opt.resize(dataset.n_images, AdamOptimizer<ArrayXf>(1e-4f, zero));
    for (uint32_t i = 0; i < dataset.n_images; ++i) {
        Eigen::Vector3f light_dir = warp_direction(dataset.metadata[i].light_dir.normalized());
        extra_dims_opt[i].reset_state(zero);
        Eigen::ArrayXf &optimzer_value = extra_dims_opt[i].variable();
        for (uint32_t j = 0; j < n_extra_dims; ++j) {
            if (dataset.has_light_dirs && j < 3)
                dst[j] = light_dir[j];
            else
                dst[j] = random_val(rng) * 2.f - 1.f;
            optimzer_value[j] = dst[j];
        }
        dst += n_extra_dims;
    }
    extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);
}

void Testbed::Nerf::Training::update_transforms(int first, int last) {
    if (last < 0) {
        last=dataset.n_images;
    }

    if (last > dataset.n_images) {
        last = dataset.n_images;
    }

    int n = last - first;
    if (n <= 0) {
        return;
    }

    if (transforms.size() < last) {
        transforms.resize(last);
    }

    for (uint32_t i = 0; i < n; ++i) {
        auto xform = dataset.xforms[i + first];
        Vector3f rot = cam_rot_offset[i + first].variable();
        float angle = rot.norm();
        rot /= angle;

        if (angle > 0) {
            xform.start.block<3, 3>(0, 0) = AngleAxisf(angle, rot) * xform.start.block<3, 3>(0, 0);
            xform.end.block<3, 3>(0, 0) = AngleAxisf(angle, rot) * xform.end.block<3, 3>(0, 0);
        }

        xform.start.col(3) += cam_pos_offset[i + first].variable();
        xform.end.col(3) += cam_pos_offset[i + first].variable();
        transforms[i + first] = xform;
    }

    transforms_gpu.enlarge(last);
    CUDA_CHECK(cudaMemcpy(transforms_gpu.data() + first, transforms.data() + first, n * sizeof(TrainingXForm), cudaMemcpyHostToDevice));
}

void Testbed::Nerf::Training::reset_camera_extrinsics() {
    for (auto&& opt : cam_rot_offset) {
        opt.reset_state();
    }

    for (auto&& opt : cam_pos_offset) {
        opt.reset_state();
    }

    for (auto&& opt : cam_exposure) {
        opt.reset_state();
    }
}

// == Testbed::NerfTracer ==============================================================================================

void Testbed::NerfTracer::init_rays_from_camera(
        uint32_t sample_index,
        uint32_t padded_output_width,
        uint32_t n_extra_dims,
        const Vector2i& resolution,
        const Vector2f& focal_length,
        const Matrix<float, 3, 4>& camera_matrix0,
        const Matrix<float, 3, 4>& camera_matrix1,
        const Vector4f& rolling_shutter,
        const Vector2f& screen_center,
        const Vector3f& parallax_shift,
        const Vector2i& quilting_dims,
        bool snap_to_pixel_centers,
        const BoundingBox& render_aabb,
        const Matrix3f& render_aabb_to_local,
        const Eigen::Matrix4f& model_matrix,
        float near_distance,
        float plane_z,
        float aperture_size,
        const Lens& lens,
        const float* envmap_data,
        const Vector2i& envmap_resolution,
        const float* distortion_data,
        const Vector2i& distortion_resolution,
        Eigen::Array4f* frame_buffer,
        float* depth_buffer,
        uint8_t* grid,
        int show_accel,
        float cone_angle_constant,
        cudaStream_t stream
) {
    // Make sure we have enough memory reserved to render at the requested resolution
    size_t n_pixels = (size_t)resolution.x() * resolution.y();
    enlarge(n_pixels, padded_output_width, n_extra_dims, stream);

    const dim3 threads = { 16, 8, 1 };
    const dim3 blocks = { div_round_up((uint32_t)resolution.x(), threads.x), div_round_up((uint32_t)resolution.y(), threads.y), 1 };
    init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(
            sample_index,
            m_rays[0].payload,
            resolution,
            focal_length,
            camera_matrix0,
            camera_matrix1,
            rolling_shutter,
            screen_center,
            parallax_shift,
            snap_to_pixel_centers,
            render_aabb,
            model_matrix,
            near_distance,
            plane_z,
            aperture_size,
            lens,
            envmap_data,
            envmap_resolution,
            frame_buffer,
            depth_buffer,
            distortion_data,
            distortion_resolution,
            quilting_dims
    );

    m_n_rays_initialized = resolution.x() * resolution.y();

    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(Array4f), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

	//RaysNerfSoa& rays_current = m_rays[0];
	//linear_kernel(generate_intersection, 0, stream,
	//	m_n_rays_initialized,
	//	render_aabb,
	//	render_aabb_to_local,
	//	focal_length,
	//	camera_matrix0.col(2),
	//	rays_current.payload,
	//	grid,
	//	(show_accel>=0) ? show_accel : 0,
	//	cone_angle_constant
	//);


    linear_kernel(advance_pos_nerf, 0, stream,
                  m_n_rays_initialized,
                  render_aabb,
                  render_aabb_to_local,
                  camera_matrix1.col(2),
                  focal_length,
                  sample_index,
                  m_rays[0].payload,
                  grid,
                  (show_accel >= 0) ? show_accel : 0,
                  cone_angle_constant
    );
}

uint32_t Testbed::NerfTracer::collide(
        int num_rays,
        NerfNetwork<network_precision_t>& network,
        const BoundingBox& render_aabb,
        const Eigen::Matrix3f& render_aabb_to_local,
        const BoundingBox& train_aabb,
        float cone_angle_constant,
        const uint8_t* grid,
        const float* extra_dims_gpu,
        ENerfActivation density_activation,
        float* collision_distances,
        cudaStream_t stream
) {

    //uint32_t n_alive = m_n_rays_initialized;
    // m_n_rays_initialized = 0;

    uint32_t i = 1;
    while (i < MARCH_ITER) {
        RaysNerfSoa& rays_current = m_rays[0];

        uint32_t n_steps_between_compaction = 8;

        uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
        PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
        linear_kernel(generate_next_nerf_network_inputs, 0, stream,
                      num_rays,
                      render_aabb,
                      render_aabb_to_local,
                      train_aabb,
                      Eigen::Vector2f{ 0.f, 0.f },
                      Eigen::Vector3f{0.f, 0.f, 0.f },
                      rays_current.payload,
                      input_data,
                      n_steps_between_compaction,
                      grid,
                      0,
                      cone_angle_constant,
                      extra_dims_gpu
        );
        uint32_t n_elements = next_multiple(num_rays * n_steps_between_compaction, tcnn::batch_size_granularity);
        GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
        GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);
        network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);


        // check collision
        CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter.data(), 0, sizeof(uint32_t), stream));

        linear_kernel(check_collision, 0, stream,
                    num_rays,
                    n_elements,
                    i,
                    train_aabb,
                    rays_current.payload,
                    input_data,
                    m_network_output,
                    network.padded_output_width(),
                    n_steps_between_compaction,
                    density_activation,
                    collision_distances,
                    m_alive_counter.data()
        );
        uint32_t n_alive;
        CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        i += n_steps_between_compaction;

        if (n_alive == 0)
            break;
    }

    return 0;
}


uint32_t Testbed::NerfTracer::intersects(
        int num_points,
        NerfNetwork<network_precision_t>& network,
        const BoundingBox& train_aabb,
        const float* extra_dims_gpu,
        ENerfActivation density_activation,
        const uint8_t* grid,
        float* intersection_densities,
        cudaStream_t stream
) {    
    RaysNerfSoa& rays_current = m_rays[0];

    uint32_t n_steps_between_compaction = 1;

    uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
    PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);


    parallel_for_gpu(num_points, [input_data, payloads=rays_current.payload, train_aabb, extra_dims_gpu] __device__ (size_t i) {
        auto& payload = payloads[i];
        input_data(i)->set_with_optional_extra_dims(warp_position(payload.origin, train_aabb), warp_direction(payload.dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims_gpu, input_data.stride_in_bytes); // XXXCONE
    });


    uint32_t n_elements = next_multiple(num_points * n_steps_between_compaction, tcnn::batch_size_granularity);
    GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
    GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);
    network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);


    parallel_for_gpu(num_points, [network_output=m_network_output, network_input=input_data, aabb=train_aabb, stride=n_elements, density_activation, intersection_densities, grid] __device__ (size_t i) {
        float network_density = network_output[i + 3 * stride];
        float dt = MIN_CONE_STEPSIZE();
        float alpha = 1.f - __expf(-network_to_density(network_density, density_activation) * dt);
        const NerfCoordinate* input = network_input(i);
        Vector3f warped_pos = input->pos.p;
        Vector3f pos = unwarp_position(warped_pos, aabb);
        uint32_t mip = max(0, mip_from_dt(dt, pos));
        if (density_grid_occupied_at(pos, grid, mip))
            intersection_densities[i] = alpha;
    });

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    return 0;
}

uint32_t Testbed::NerfTracer::trace(
        NerfNetwork<network_precision_t>& network,
        const BoundingBox& render_aabb,
        const Eigen::Matrix3f& render_aabb_to_local,
        const BoundingBox& train_aabb,
        const uint32_t n_training_images,
        const TrainingXForm* training_xforms,
        const Vector2f& focal_length,
        float cone_angle_constant,
        const uint8_t* grid,
        const Eigen::Matrix<float, 3, 4> &camera_matrix,
        float depth_scale,
        int visualized_layer,
        int visualized_dim,
        ENerfActivation rgb_activation,
        ENerfActivation density_activation,
        int show_accel,
        float min_transmittance,
        float glow_y_cutoff,
        int glow_mode,
        const float* extra_dims_gpu,
        cudaStream_t stream
) {
    if (m_n_rays_initialized == 0) {
        return 0;
    }
    
    CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter.data(), 0, sizeof(uint32_t), stream));

    uint32_t n_alive = m_n_rays_initialized;
    // m_n_rays_initialized = 0;


    uint32_t i = 1;
    uint32_t double_buffer_index = 0;
    while (i < MARCH_ITER) {
        RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
        RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
        ++double_buffer_index;

        // Compact rays that did not diverge yet
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter.data(), 0, sizeof(uint32_t), stream));
            linear_kernel(compact_kernel_nerf, 0, stream,
                          n_alive,
                          rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
                          rays_current.rgba, rays_current.depth, rays_current.payload,
                          m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
                          m_alive_counter.data(), m_hit_counter.data()
            );
            CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        }

        if (n_alive == 0) {
            break;
        }

        uint32_t n_steps_between_compaction = tcnn::clamp(m_n_rays_initialized / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

        uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
        PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
        linear_kernel(generate_next_nerf_network_inputs, 0, stream,
                      n_alive,
                      render_aabb,
                      render_aabb_to_local,
                      train_aabb,
                      focal_length,
                      camera_matrix.col(2),
                      rays_current.payload,
                      input_data,
                      n_steps_between_compaction,
                      grid,
                      (show_accel>=0) ? show_accel : 0,
                      cone_angle_constant,
                      extra_dims_gpu
        );
        uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, tcnn::batch_size_granularity);
        GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
        GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);
        network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

        linear_kernel(composite_kernel_nerf, 0, stream,
                      n_alive,
                      n_elements,
                      i,
                      train_aabb,
                      glow_y_cutoff,
                      glow_mode,
                      n_training_images,
                      training_xforms,
                      camera_matrix,
                      focal_length,
                      depth_scale,
                      rays_current.rgba,
                      rays_current.depth,
                      rays_current.payload,
                      input_data,
                      m_network_output,
                      network.padded_output_width(),
                      n_steps_between_compaction,
                      grid,
                      rgb_activation,
                      density_activation,
                      show_accel,
                      min_transmittance
        );

        i += n_steps_between_compaction;
    }

    uint32_t n_hit;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return n_hit;
}

void Testbed::NerfTracer::enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
    n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
    size_t num_floats = sizeof(NerfCoordinate) / 4 + n_extra_dims;
    auto scratch = allocate_workspace_and_distribute<
            Array4f, float, NerfPayload, // m_rays[0]
            Array4f, float, NerfPayload, // m_rays[1]
            Array4f, float, NerfPayload, // m_rays_hit

            network_precision_t,
            float
    >(
            stream, &m_scratch_alloc,
            n_elements, n_elements, n_elements,
            n_elements, n_elements, n_elements,
            n_elements, n_elements, n_elements,
            n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
            n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats
    );

    m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
    m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
    m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

    m_network_output = std::get<9>(scratch);
    m_network_input = std::get<10>(scratch);
}

NGP_NAMESPACE_END