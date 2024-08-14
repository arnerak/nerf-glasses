#pragma once

#include "../common.h"

#include <Eigen/Dense>

//#include <tinygltf/stb_image.h>

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <cuda_fp16.h>
#include <vector_types.h>

#include <cstdint>
#include <string>
#include <chrono>
#include <functional>

#include "random_val.cuh"   // TODO check if this works out

// == Instant-NGP ======================================================================================================

NGP_NAMESPACE_BEGIN

using precision_t = tcnn::network_precision_t;

//using json = nlohmann::json;

struct Ray {
    Eigen::Vector3f o;
    Eigen::Vector3f d;
};

struct TrainingXForm {
    Eigen::Matrix<float, 3, 4> start;
    Eigen::Matrix<float, 3, 4> end;
};

enum class ELensMode : int {
    Perspective,
    OpenCV,
    FTheta,
    LatLong
};

struct Lens {
    ELensMode mode = ELensMode::Perspective;
    float params[7] = {};
};

enum class EImageDataType {
    None,
    Byte,
    Half,
    Float
};

enum class EDepthDataType {
    UShort,
    Float
};

enum class EColorSpace : int {
    Linear,
    SRGB,
    VisPosNeg
};

enum class ETonemapCurve : int {
    Identity,
    ACES,
    Hable,
    Reinhard
};

enum class EGroundTruthRenderMode : int {
    Shade,
    Depth,
    NumRenderModes,
};

enum class ENerfActivation : int {
    None,
    ReLU,
    Logistic,
    Exponential,
};
static constexpr const char* NerfActivationStr = "None\0ReLU\0Logistic\0Exponential\0\0"; // For ImGui

enum class ELossType : int {
    L2,
    L1,
    Mape,
    Smape,
    Huber,
    LogL1,
    RelativeL2,
};
static constexpr const char* LossTypeStr = "L2\0L1\0MAPE\0SMAPE\0Huber\0LogL1\0RelativeL2\0\0"; // For ImGui


inline std::string replace_all(std::string str, const std::string& a, const std::string& b) {
    std::string::size_type n = 0;
    while ((n = str.find(a, n)) != std::string::npos) {
        str.replace(n, a.length(), b);
        n += b.length();
    }
    return str;
}

inline HOST_DEVICE float sign(float x) {
    return copysignf(1.0, x);
}

inline HOST_DEVICE float4 to_float4(const Eigen::Array4f& x) {
    return {x.x(), x.y(), x.z(), x.w()};
}

inline HOST_DEVICE float fov_to_focal_length(int resolution, float degrees) {
    return 0.5f * (float)resolution / tanf(0.5f * degrees * NMR_PI/180.0f);
}

inline HOST_DEVICE float linear_to_srgb(float linear) {
    if (linear < 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
    }
}

inline HOST_DEVICE Eigen::Array3f linear_to_srgb(const Eigen::Array3f& x) {
    return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline HOST_DEVICE float srgb_to_linear(float srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    } else {
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

inline HOST_DEVICE Eigen::Array3f srgb_to_linear(const Eigen::Array3f& x) {
    return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

inline HOST_DEVICE Eigen::Vector2i image_pos(const Eigen::Vector2f& pos, const Eigen::Vector2i& resolution) {
    return pos.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMin(resolution - Eigen::Vector2i::Constant(1)).cwiseMax(0);
}

inline HOST_DEVICE uint64_t pixel_idx(const Eigen::Vector2i& pos, const Eigen::Vector2i& resolution, uint32_t img) {
    return pos.x() + pos.y() * resolution.x() + img * (uint64_t)resolution.x() * resolution.y();
}

inline HOST_DEVICE uint64_t pixel_idx(const Eigen::Vector2f& xy, const Eigen::Vector2i& resolution, uint32_t img) {
    return pixel_idx(image_pos(xy, resolution), resolution, img);
}

inline HOST_DEVICE Eigen::Array4f read_rgba(Eigen::Vector2i px, const Eigen::Vector2i& resolution, const void* pixels, EImageDataType image_data_type, uint32_t img = 0) {
    switch (image_data_type) {
        default:
            // This should never happen. Bright red to indicate this.
            return Eigen::Array4f{5.0f, 0.0f, 0.0f, 1.0f};
        case EImageDataType::Byte: {
            uint8_t val[4];
            *(uint32_t*)&val[0] = ((uint32_t*)pixels)[pixel_idx(px, resolution, img)];
            if (*(uint32_t*)&val[0] == 0x00FF00FF) {
                return Eigen::Array4f::Constant(-1.0f);
            }

            float alpha = (float)val[3] * (1.0f/255.0f);
            return Eigen::Array4f{
                    srgb_to_linear((float)val[0] * (1.0f/255.0f)) * alpha,
                    srgb_to_linear((float)val[1] * (1.0f/255.0f)) * alpha,
                    srgb_to_linear((float)val[2] * (1.0f/255.0f)) * alpha,
                    alpha,
            };
        }
        case EImageDataType::Half: {
            __half val[4];
            *(uint64_t*)&val[0] = ((uint64_t*)pixels)[pixel_idx(px, resolution, img)];
            return Eigen::Array4f{val[0], val[1], val[2], val[3]};
        }
        case EImageDataType::Float:
            return ((Eigen::Array4f*)pixels)[pixel_idx(px, resolution, img)];
    }
}

template <typename T>
__global__ void from_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out, bool white_2_transparent = false, bool black_2_transparent = false, uint32_t mask_color = 0) {
    const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_pixels) return;

    uint8_t rgba[4];
    *((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

    float alpha = rgba[3] * (1.0f/255.0f);
    // NSVF dataset has 'white = transparent' madness
    if (white_2_transparent && rgba[0]==255 && rgba[1]==255 && rgba[2]==255) {
        alpha = 0.f;
    }
    if (black_2_transparent && rgba[0]==0 && rgba[1]==0 && rgba[2]==0) {
        alpha = 0.f;
    }

    tcnn::vector_t<T, 4> rgba_out;
    rgba_out[0] = (T)(srgb_to_linear(rgba[0] * (1.0f/255.0f)) * alpha);
    rgba_out[1] = (T)(srgb_to_linear(rgba[1] * (1.0f/255.0f)) * alpha);
    rgba_out[2] = (T)(srgb_to_linear(rgba[2] * (1.0f/255.0f)) * alpha);
    rgba_out[3] = (T)alpha;

    if (mask_color != 0 && mask_color == *((uint32_t*)&rgba[0])) {
        rgba_out[0] = rgba_out[1] = rgba_out[2] = rgba_out[3] = (T)-1.0f;
    }

    *((tcnn::vector_t<T, 4>*)&out[i*4]) = rgba_out;
}

//tcnn::GPUMemory<float> load_stbi(const std::string& filename, int& width, int& height) {
//    using namespace tcnn;
//
//    bool is_hdr = stbi_is_hdr(filename.c_str());
//
//    void* data; // width * height * RGBA
//    int comp;
//    if (is_hdr) {
//        data = stbi_loadf(filename.c_str(), &width, &height, &comp, 4);
//    } else {
//        data = stbi_load(filename.c_str(), &width, &height, &comp, 4);
//    }
//
//    if (!data) {
//        throw std::runtime_error{std::string{stbi_failure_reason()}};
//    }
//
//    ScopeGuard mem_guard{[&]() { stbi_image_free(data); }};
//
//    if (width == 0 || height == 0) {
//        throw std::runtime_error{"Image has zero pixels."};
//    }
//
//    GPUMemory<float> result(width * height * 4);
//    if (is_hdr) {
//        result.copy_from_host((float*)data);
//    } else {
//        GPUMemory<uint8_t> bytes(width * height * 4);
//        bytes.copy_from_host((uint8_t*)data);
//        linear_kernel(from_rgba32<float>, 0, nullptr, width * height, bytes.data(), result.data(), false, false, 0);
//    }
//
//    return result;
//}

inline HOST_DEVICE void apply_quilting(uint32_t* x, uint32_t* y, const Eigen::Vector2i& resolution, Eigen::Vector3f& parallax_shift, const Eigen::Vector2i& quilting_dims) {
    float resx = float(resolution.x()) / quilting_dims.x();
    float resy = float(resolution.y()) / quilting_dims.y();
    int panelx = (int)floorf(*x/resx);
    int panely = (int)floorf(*y/resy);
    *x = (*x - panelx * resx);
    *y = (*y - panely * resy);
    int idx = panelx + quilting_dims.x() * panely;

    if (quilting_dims == Eigen::Vector2i{2, 1}) {
        // Likely VR: parallax_shift.x() is the IPD in this case. The following code centers the camera matrix between both eyes.
        parallax_shift.x() = idx ? (-0.5f * parallax_shift.x()) : (0.5f * parallax_shift.x());
    } else {
        // Likely HoloPlay lenticular display: in this case, `parallax_shift.z()` is the inverse height of the head above the display.
        // The following code computes the x-offset of views as a function of this.
        const float max_parallax_angle = 17.5f; // suggested value in https://docs.lookingglassfactory.com/keyconcepts/camera
        float parallax_angle = max_parallax_angle * PI() / 180.f * ((idx+0.5f)*2.f / float(quilting_dims.y() * quilting_dims.x()) - 1.f);
        parallax_shift.x() = atanf(parallax_angle) / parallax_shift.z();
    }
}

inline HOST_DEVICE Eigen::Vector3f f_theta_undistortion(const Eigen::Vector2f& uv, const float* params, const Eigen::Vector3f& error_direction) {
    // we take f_theta intrinsics to be: r0, r1, r2, r3, resx, resy; we rescale to whatever res the intrinsics specify.
    float xpix = uv.x() * params[5];
    float ypix = uv.y() * params[6];
    float norm = sqrtf(xpix*xpix + ypix*ypix);
    float alpha = params[0] + norm * (params[1] + norm * (params[2] + norm * (params[3] + norm * params[4])));
    float sin_alpha, cos_alpha;
    sincosf(alpha, &sin_alpha, &cos_alpha);
    if (cos_alpha <= std::numeric_limits<float>::min() || norm == 0.f) {
        return error_direction;
    }
    sin_alpha *= 1.f / norm;
    return { sin_alpha * xpix, sin_alpha * ypix, cos_alpha };
}

inline HOST_DEVICE Eigen::Vector3f latlong_to_dir(const Eigen::Vector2f& uv) {
    float theta = (uv.y() - 0.5f) * PI();
    float phi = (uv.x() - 0.5f) * PI() * 2.0f;
    float sp, cp, st, ct;
    sincosf(theta, &st, &ct);
    sincosf(phi, &sp, &cp);
    return {sp * ct, st, cp * ct};
}

template <uint32_t N_DIMS, typename T>
HOST_DEVICE Eigen::Matrix<float, N_DIMS, 1> read_image(const T* __restrict__ data, const Eigen::Vector2i& resolution, const Eigen::Vector2f& pos) {
    const Eigen::Vector2f pos_float = Eigen::Vector2f{pos.x() * (float)(resolution.x()-1), pos.y() * (float)(resolution.y()-1)};
    const Eigen::Vector2i texel = pos_float.cast<int>();

    const Eigen::Vector2f weight = pos_float - texel.cast<float>();

    auto read_val = [&](Eigen::Vector2i pos) {
        pos.x() = std::max(std::min(pos.x(), resolution.x()-1), 0);
        pos.y() = std::max(std::min(pos.y(), resolution.y()-1), 0);

        Eigen::Matrix<float, N_DIMS, 1> result;
        if (std::is_same<T, float>::value) {
            result = *(Eigen::Matrix<T, N_DIMS, 1>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];
        } else {
            auto val = *(tcnn::vector_t<T, N_DIMS>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];

            PRAGMA_UNROLL
            for (uint32_t i = 0; i < N_DIMS; ++i) {
                result[i] = (float)val[i];
            }
        }
        return result;
    };

    return (
            (1 - weight.x()) * (1 - weight.y()) * read_val({texel.x(), texel.y()}) +
            (weight.x()) * (1 - weight.y()) * read_val({texel.x()+1, texel.y()}) +
            (1 - weight.x()) * (weight.y()) * read_val({texel.x(), texel.y()+1}) +
            (weight.x()) * (weight.y()) * read_val({texel.x()+1, texel.y()+1})
    );
}

inline HOST_DEVICE Ray pixel_to_ray(
        uint32_t spp,
        const Eigen::Vector2i& pixel,
        const Eigen::Vector2i& resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Matrix<float, 3, 4>& camera_matrix,
        const Eigen::Vector2f& screen_center,
        const Eigen::Vector3f& parallax_shift,
        bool snap_to_pixel_centers = false,
        float near_distance = 0.0f,
        float focus_z = 1.0f,
        float aperture_size = 0.0f,
        const Lens& lens = {},
        const float* __restrict__ distortion_grid = nullptr,
        const Eigen::Vector2i distortion_grid_resolution = Eigen::Vector2i::Zero()
) {
    Eigen::Vector2f offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : spp);
    Eigen::Vector2f uv = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());

    Eigen::Vector3f dir;
    if (lens.mode == ELensMode::FTheta) {
        dir = f_theta_undistortion(uv - screen_center, lens.params, {1000.f, 0.f, 0.f});
        if (dir.x() == 1000.f) {
            return {{1000.f, 0.f, 0.f}, {0.f, 0.f, 1.f}}; // return a point outside the aabb so the pixel is not rendered
        }
    } else if (lens.mode == ELensMode::LatLong) {
        dir = latlong_to_dir(uv);
    } else {
        dir = {
//                (uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
//                (uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
                2.0f * ((static_cast<float>(pixel.x()) + 0.5f) / static_cast<float>(resolution.x())) - 1.0f,
                2.0f * ((static_cast<float>(pixel.y()) + 0.5f) / static_cast<float>(resolution.y())) - 1.0f,
                1.0f
        };
        if (lens.mode == ELensMode::OpenCV) {
            // TODO kebiro: iterative_opencv_lens_undistortion() ... LensMode needed at all?
//            iterative_opencv_lens_undistortion(lens.params, &dir.x(), &dir.y());
        }
    }
    if (distortion_grid) {
        dir.head<2>() += read_image<2>(distortion_grid, distortion_grid_resolution, uv);
    }

    Eigen::Vector3f head_pos = {parallax_shift.x(), parallax_shift.y(), 0.f};
    dir -= head_pos * parallax_shift.z(); // we could use focus_z here in the denominator. for now, we pack m_scale in here.
    dir = camera_matrix.block<3, 3>(0, 0) * dir;

    Eigen::Vector3f origin = camera_matrix.block<3, 3>(0, 0) * head_pos + camera_matrix.col(3);

    if (aperture_size > 0.0f) {
        Eigen::Vector3f lookat = origin + dir * focus_z;
        Eigen::Vector2f blur = aperture_size * square2disk_shirley(ld_random_val_2d(spp, (uint32_t)pixel.x() * 19349663 + (uint32_t)pixel.y() * 96925573) * 2.0f - Eigen::Vector2f::Ones());
        origin += camera_matrix.block<3, 2>(0, 0) * blur;
        dir = (lookat - origin) / focus_z;
    }

    origin += dir * near_distance;

    return {origin, dir};
}

enum class EEmaType {
    Time,
    Step,
};

class Ema {
public:
    Ema(EEmaType type, float half_life)
            : m_type{type}, m_decay{std::pow(0.5f, 1.0f / half_life)}, m_creation_time{std::chrono::steady_clock::now()} {}

    int64_t current_progress() {
        if (m_type == EEmaType::Time) {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_creation_time).count();
        } else {
            return m_last_progress + 1;
        }
    }

    void update(float val) {
        int64_t cur = current_progress();
        int64_t elapsed = cur - m_last_progress;
        m_last_progress = cur;

        float decay = std::pow(m_decay, elapsed);
        m_val = val;
        m_ema_val = decay * m_ema_val + (1.0f - decay) * val;
    }

    void set(float val) {
        m_last_progress = current_progress();
        m_val = m_ema_val = val;
    }

    float val() const {
        return m_val;
    }

    float ema_val() const {
        return m_ema_val;
    }

private:
    float m_val = 0.0f;
    float m_ema_val = 0.0f;
    EEmaType m_type;
    float m_decay;

    int64_t m_last_progress = 0;
    std::chrono::time_point<std::chrono::steady_clock> m_creation_time;
};

NGP_NAMESPACE_END