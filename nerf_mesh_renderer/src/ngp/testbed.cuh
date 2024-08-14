/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

//#include <neural-graphics-primitives/discrete_distribution.h>
//#include <neural-graphics-primitives/shared_queue.h>
//#include <neural-graphics-primitives/trainable_buffer.cuh>

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <json/json.hpp>

#include <filesystem/path.h>

#ifdef NGP_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

#include <thread>

#include "adam_optimizer.h"
#include "nerf.cuh"
#include "nerf_loader.cuh"
#include "ngp_common.cuh"
#include "random_val.cuh"
#include "render_buffer.cuh"
#include "trainable_buffer.cuh"

TCNN_NAMESPACE_BEGIN
    template <typename T> class Loss;
    template <typename T> class Optimizer;
    template <typename T> class Encoding;
    template <typename T, typename PARAMS_T> class Network;
    template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
TCNN_NAMESPACE_END

NGP_NAMESPACE_BEGIN

    template <typename T> class NerfNetwork;

    class GLTexture;

    class Testbed {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        explicit Testbed(std::string name);

        std::string m_name; // From kebiro.

        class NerfTracer {
        public:
            NerfTracer() : m_hit_counter(1), m_alive_counter(1) {}

            void init_rays_from_camera(
                    uint32_t spp,
                    uint32_t padded_output_width,
                    uint32_t n_extra_dims,
                    const Eigen::Vector2i& resolution,
                    const Eigen::Vector2f& focal_length,
                    const Eigen::Matrix<float, 3, 4>& camera_matrix0,
                    const Eigen::Matrix<float, 3, 4>& camera_matrix1,
                    const Eigen::Vector4f& rolling_shutter,
                    const Eigen::Vector2f& screen_center,
                    const Eigen::Vector3f& parallax_shift,
                    const Eigen::Vector2i& quilting_dims,
                    bool snap_to_pixel_centers,
                    const BoundingBox& render_aabb,
                    const Eigen::Matrix3f& render_aabb_to_local,
                    const Eigen::Matrix4f& model_matrix,
                    float near_distance,
                    float plane_z,
                    float aperture_size,
                    const Lens& lens,
                    const float* envmap_data,
                    const Eigen::Vector2i& envmap_resolution,
                    const float* distortion_data,
                    const Eigen::Vector2i& distortion_resolution,
                    Eigen::Array4f* frame_buffer,
                    float* depth_buffer,
                    uint8_t* grid,
                    int show_accel,
                    float cone_angle_constant,
                    cudaStream_t stream
            );


            uint32_t collide(
                    int num_rays,
                    NerfNetwork<precision_t>& network,
                    const BoundingBox& render_aabb,
                    const Eigen::Matrix3f& render_aabb_to_local,
                    const BoundingBox& train_aabb,
                    float cone_angle_constant,
                    const uint8_t* grid,
                    const float* extra_dims_gpu,
                    ENerfActivation density_activation,
                    float* collision_distances,
                    cudaStream_t stream
            );

            uint32_t intersects(
                int num_points,
                NerfNetwork<precision_t>& network,
                const BoundingBox& train_aabb,
                const float* extra_dims_gpu,
                ENerfActivation density_activation,
                const uint8_t* grid,
                float* intersection_densities,
                cudaStream_t stream
            );

            uint32_t trace(
                NerfNetwork<precision_t>& network,
                const BoundingBox& render_aabb,
                const Eigen::Matrix3f& render_aabb_to_local,
                const BoundingBox& train_aabb,
                const uint32_t n_training_images,
                const TrainingXForm* training_xforms,
                const Eigen::Vector2f& focal_length,
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
            );

            void enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream);
            RaysNerfSoa& rays_hit() { return m_rays_hit; }
            RaysNerfSoa& rays_init() { return m_rays[0]; }
            uint32_t n_rays_initialized() const { return m_n_rays_initialized; }

            void clear() {
                m_scratch_alloc = {};
            }

        private:
            RaysNerfSoa m_rays[2];
            RaysNerfSoa m_rays_hit;
            precision_t* m_network_output;
            float* m_network_input;
            tcnn::GPUMemory<uint32_t> m_hit_counter;
            tcnn::GPUMemory<uint32_t> m_alive_counter;
            uint32_t m_n_rays_initialized = 0;
            tcnn::GPUMemoryArena::Allocation m_scratch_alloc;
        };

        struct LevelStats {
            float mean() { return count ? (x / (float)count) : 0.f; }
            float variance() { return count ? (xsquared - (x * x) / (float)count) / (float)count : 0.f; }
            float sigma() { return sqrtf(variance()); }
            float fraczero() { return (float)numzero / float(count + numzero); }
            float fracquant() { return (float)numquant / float(count); }

            float x;
            float xsquared;
            float min;
            float max;
            int numzero;
            int numquant;
            int count;
        };

        static constexpr float LOSS_SCALE = 128.f;

        struct NetworkDims {
            uint32_t n_input;
            uint32_t n_output;
            uint32_t n_pos;
        };

        NetworkDims network_dims_volume() const;
        NetworkDims network_dims_sdf() const;
        NetworkDims network_dims_image() const;
        NetworkDims network_dims_nerf() const;

        NetworkDims network_dims() const;

        const float* get_inference_extra_dims(cudaStream_t stream) const;
        void render_nerf(CudaRenderBuffer& render_buffer, const Eigen::Vector2i& max_res, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix0, const Eigen::Matrix<float, 3, 4>& camera_matrix1, const Eigen::Vector4f& rolling_shutter, const Eigen::Vector2f& screen_center, cudaStream_t stream);
        void render_frame(const Eigen::Matrix<float, 3, 4>& camera_matrix0, const Eigen::Matrix<float, 3, 4>& camera_matrix1, const Eigen::Vector4f& nerf_rolling_shutter, CudaRenderBuffer& render_buffer, bool to_srgb = true) ;
        nlohmann::json load_network_config(const filesystem::path& network_config_path);
        void reload_network_from_file(const std::string& network_config_path);
        void reload_network_from_json(const nlohmann::json& json, const std::string& config_base_path=""); // config_base_path is needed so that if the passed in json uses the 'parent' feature, we know where to look... be sure to use a filename, or if a directory, end with a trailing slash
        void reset_accumulation(bool due_to_camera_movement = false, bool immediate_redraw = true);
        void redraw_next_frame() {
            m_render_skip_due_to_lack_of_camera_movement_counter = 0;
        }
        bool reprojection_available() { /*Before: returned m_dlss */return false; }
        static ELossType string_to_loss_type(const std::string& str);
        void reset_network(bool clear_density_grid = true);
        void create_empty_nerf_dataset(size_t n_images, int aabb_scale = 1, bool is_hdr = false);
        void load_nerf();
        void load_nerf_post();
        void load_mesh();
        void set_exposure(float exposure) { m_exposure = exposure; }
        void set_max_level(float maxlevel);
        void set_min_level(float minlevel);
        void translate_camera(const Eigen::Vector3f& rel);
        void handle_file(const std::string& file);
        void set_nerf_camera_matrix(const Eigen::Matrix<float, 3, 4>& cam);
        Eigen::Vector3f look_at() const;
        void set_look_at(const Eigen::Vector3f& pos);
        float scale() const { return m_scale; }
        void set_scale(float scale);
        Eigen::Vector3f view_pos() const { return m_camera.col(3); }
        Eigen::Vector3f view_dir() const { return m_camera.col(2); }
        Eigen::Vector3f view_up() const { return m_camera.col(1); }
        Eigen::Vector3f view_side() const { return m_camera.col(0); }
        void set_view_dir(const Eigen::Vector3f& dir);
        void reset_camera();
        void update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream);
        void update_density_grid_mean_and_bitfield(cudaStream_t stream);

        struct NerfCounters {
            tcnn::GPUMemory<uint32_t> numsteps_counter; // number of steps each ray took
            tcnn::GPUMemory<uint32_t> numsteps_counter_compacted; // number of steps each ray took
            tcnn::GPUMemory<float> loss;

            uint32_t rays_per_batch = 1<<12;
            uint32_t n_rays_total = 0;
            uint32_t measured_batch_size = 0;
            uint32_t measured_batch_size_before_compaction = 0;

            void prepare_for_training_steps(cudaStream_t stream);
            float update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
        };

        void imgui();
        Eigen::Vector2f calc_focal_length(const Eigen::Vector2i& resolution, int fov_axis, float zoom) const ;
        Eigen::Vector2f render_screen_center() const ;
        tcnn::GPUMemory<Eigen::Array4f> get_rgba_on_grid(Eigen::Vector3i res3d, Eigen::Vector3f ray_dir, bool voxel_centers, float depth, bool density_as_alpha = false);

        // Determines the 3d focus point by rendering a little 16x16 depth image around
        // the mouse cursor and picking the median depth.
        void determine_autofocus_target_from_pixel(const Eigen::Vector2i& focus_pixel);
        void autofocus();
        size_t n_params();
        size_t first_encoder_param();
        size_t n_encoding_params();

#ifdef NGP_PYTHON
	pybind11::dict compute_marching_cubes_mesh(Eigen::Vector3i res3d = Eigen::Vector3i::Constant(128), BoundingBox aabb = BoundingBox{Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones()}, float thresh=2.5f);
	pybind11::array_t<float> render_to_cpu(int width, int height, int spp, bool linear);
	pybind11::array_t<float> render_with_rolling_shutter_to_cpu(const Eigen::Matrix<float, 3, 4>& camera_transform_start, const Eigen::Matrix<float, 3, 4>& camera_transform_end, const Eigen::Vector4f& rolling_shutter, int width, int height, int spp, bool linear);
	pybind11::array_t<float> screenshot(bool linear) const;
	void override_sdf_training_data(pybind11::array_t<float> points, pybind11::array_t<float> distances);
#endif

        double calculate_iou(uint32_t n_samples=128*1024*1024, float scale_existing_results_factor=0.0, bool blocking=true, bool force_use_octree = true);
        void train_and_render(bool skip_rendering);
        filesystem::path training_data_path() const;
        void init_window(int resw, int resh, bool hidden = false, bool second_window = false);
        void destroy_window();
        void apply_camera_smoothing(float elapsed_ms);
        int find_best_training_view(int default_view);
        bool begin_frame_and_handle_user_input();
        void gather_histograms();
        void draw_gui();
        bool frame();
        uint32_t n_dimensions_to_visualize() const;
        float fov() const ;
        void set_fov(float val) ;
        Eigen::Vector2f fov_xy() const ;
        void set_fov_xy(const Eigen::Vector2f& val);
        void load_snapshot(const std::string& filepath_string);
        void set_camera_from_time(float t);
        void update_loss_graph();
        void load_camera_path(const std::string& filepath_string);
        bool loop_animation();
        void set_loop_animation(bool value);

        bool m_render_window = false;
        bool m_gather_histograms = false;

        bool m_include_optimizer_state_in_snapshot = false;
        bool m_render_ground_truth = false;
        EGroundTruthRenderMode m_ground_truth_render_mode = EGroundTruthRenderMode::Shade;
        float m_ground_truth_alpha = 1.0f;

        bool m_train = false;
        bool m_training_data_available = false;
        bool m_render = true;
        int m_max_spp = 0;
        bool m_max_level_rand_training = false;

        // Rendering stuff
        Eigen::Vector2i m_window_res = Eigen::Vector2i::Constant(0);
        bool m_dynamic_res = true;
        float m_dynamic_res_target_fps = 20.0f;
        int m_fixed_res_factor = 8;
        float m_last_render_res_factor = 1.0f;
        float m_scale = 1.0;
        float m_prev_scale = 1.0;
        float m_aperture_size = 0.0f;
        Eigen::Vector2f m_relative_focal_length = Eigen::Vector2f::Ones();
        uint32_t m_fov_axis = 1;
        float m_zoom = 1.f; // 2d zoom factor (for insets?)
        Eigen::Vector2f m_screen_center = Eigen::Vector2f::Constant(0.5f); // center of 2d zoom

        Eigen::Matrix<float, 3, 4> m_camera = Eigen::Matrix<float, 3, 4>::Zero();
        Eigen::Matrix<float, 3, 4> m_smoothed_camera = Eigen::Matrix<float, 3, 4>::Zero();
        Eigen::Matrix<float, 3, 4> m_prev_camera = Eigen::Matrix<float, 3, 4>::Zero();
        size_t m_render_skip_due_to_lack_of_camera_movement_counter = 0;

        bool m_fps_camera = false;
        bool m_camera_smoothing = false;
        bool m_autofocus = false;
        Eigen::Vector3f m_autofocus_target = Eigen::Vector3f::Constant(0.5f);

        Eigen::Vector3f m_up_dir = {0.0f, 1.0f, 0.0f};
        Eigen::Vector3f m_sun_dir = Eigen::Vector3f::Ones().normalized();
        float m_bounding_radius = 1;
        float m_exposure = 0.f;

        Eigen::Vector2i m_quilting_dims = Eigen::Vector2i::Ones();

        uint32_t m_seed = 1337;

        std::shared_ptr<GLTexture> m_pip_render_texture;
        std::vector<std::shared_ptr<GLTexture>> m_render_textures;

        std::vector<CudaRenderBuffer> m_render_surfaces;
        std::unique_ptr<CudaRenderBuffer> m_pip_render_surface;

        void redraw_gui_next_frame() {
            m_gui_redraw = true;
        }

        bool m_gui_redraw = true;

        struct Nerf {
            NerfTracer tracer;

            struct Training {
                NerfDataset dataset;
                int n_images_for_training = 0; // how many images to train from, as a high watermark compared to the dataset size
                int n_images_for_training_prev = 0; // how many images we saw last time we updated the density grid

                struct ErrorMap {
                    tcnn::GPUMemory<float> data;
                    tcnn::GPUMemory<float> cdf_x_cond_y;
                    tcnn::GPUMemory<float> cdf_y;
                    tcnn::GPUMemory<float> cdf_img;
                    std::vector<float> pmf_img_cpu;
                    Eigen::Vector2i resolution = {16, 16};
                    Eigen::Vector2i cdf_resolution = {16, 16};
                    bool is_cdf_valid = false;
                } error_map;

                std::vector<TrainingXForm> transforms;
                tcnn::GPUMemory<TrainingXForm> transforms_gpu;

                std::vector<Eigen::Vector3f> cam_pos_gradient;
                tcnn::GPUMemory<Eigen::Vector3f> cam_pos_gradient_gpu;

                std::vector<Eigen::Vector3f> cam_rot_gradient;
                tcnn::GPUMemory<Eigen::Vector3f> cam_rot_gradient_gpu;

                tcnn::GPUMemory<Eigen::Array3f> cam_exposure_gpu;
                std::vector<Eigen::Array3f> cam_exposure_gradient;
                tcnn::GPUMemory<Eigen::Array3f> cam_exposure_gradient_gpu;

                Eigen::Vector2f cam_focal_length_gradient = Eigen::Vector2f::Zero();
                tcnn::GPUMemory<Eigen::Vector2f> cam_focal_length_gradient_gpu;

                std::vector<AdamOptimizer<Eigen::Array3f>> cam_exposure;
                std::vector<AdamOptimizer<Eigen::Vector3f>> cam_pos_offset;
                std::vector<RotationAdamOptimizer> cam_rot_offset;
                AdamOptimizer<Eigen::Vector2f> cam_focal_length_offset = AdamOptimizer<Eigen::Vector2f>(0.f);

                tcnn::GPUMemory<float> extra_dims_gpu; // if the model demands a latent code per training image, we put them in here.
                tcnn::GPUMemory<float> extra_dims_gradient_gpu;
                std::vector<AdamOptimizer<Eigen::ArrayXf>> extra_dims_opt;

                void reset_extra_dims(default_rng_t &rng);

                float extrinsic_l2_reg = 1e-4f;
                float extrinsic_learning_rate = 1e-3f;

                float intrinsic_l2_reg = 1e-4f;
                float exposure_l2_reg = 0.0f;

                NerfCounters counters_rgb;

                bool random_bg_color = true;
                bool linear_colors = false;
                ELossType loss_type = ELossType::L2;
                ELossType depth_loss_type = ELossType::L1;
                bool snap_to_pixel_centers = true;
                bool train_envmap = false;

                bool optimize_distortion = false;
                bool optimize_extrinsics = false;
                bool optimize_extra_dims = false;
                bool optimize_focal_length = false;
                bool optimize_exposure = false;
                bool render_error_overlay = false;
                float error_overlay_brightness = 0.125f;
                uint32_t n_steps_between_cam_updates = 16;
                uint32_t n_steps_since_cam_update = 0;

                bool sample_focal_plane_proportional_to_error = false;
                bool sample_image_proportional_to_error = false;
                bool include_sharpness_in_error = false;
                uint32_t n_steps_between_error_map_updates = 128;
                uint32_t n_steps_since_error_map_update = 0;
                uint32_t n_rays_since_error_map_update = 0;

                float near_distance = 0.2f;
                float density_grid_decay = 0.95f;
                default_rng_t density_grid_rng;
                int view = 0;

                float depth_supervision_lambda = 0.f;

                tcnn::GPUMemory<float> sharpness_grid;

                void set_camera_intrinsics(int frame_idx, float fx, float fy = 0.0f, float cx = -0.5f, float cy = -0.5f, float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f);
                void set_camera_extrinsics_rolling_shutter(int frame_idx, Eigen::Matrix<float, 3, 4> camera_to_world_start, Eigen::Matrix<float, 3, 4> camera_to_world_end, const Eigen::Vector4f& rolling_shutter, bool convert_to_ngp = true);
                void set_camera_extrinsics(int frame_idx, Eigen::Matrix<float, 3, 4> camera_to_world, bool convert_to_ngp = true);
                Eigen::Matrix<float, 3, 4> get_camera_extrinsics(int frame_idx);
                void update_transforms(int first = 0, int last = -1);

#ifdef NGP_PYTHON
                void set_image(int frame_idx, pybind11::array_t<float> img, pybind11::array_t<float> depth_img, float depth_scale);
#endif

                void reset_camera_extrinsics();
                void export_camera_extrinsics(const std::string& filename, bool export_extrinsics_in_quat_format = true);
            } training = {};

            static uint32_t pos_to_cascaded_grid_idx(Eigen::Vector3f pos, uint32_t mip);
            tcnn::GPUMemory<float> density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
            tcnn::GPUMemory<uint8_t> density_grid_bitfield;
            uint8_t* get_density_grid_bitfield_mip(uint32_t mip);
            tcnn::GPUMemory<float> density_grid_mean;
            uint32_t density_grid_ema_step = 0;

            uint32_t max_cascade = 0;

            tcnn::GPUMemory<float> vis_input;
            tcnn::GPUMemory<Eigen::Array4f> vis_rgba;

            ENerfActivation rgb_activation = ENerfActivation::Exponential;
            ENerfActivation density_activation = ENerfActivation::Exponential;

            Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.5f);
            uint32_t extra_dim_idx_for_inference = 0; // which training image's latent code should be presented at inference time

            int show_accel = -1;

            float sharpen = 0.f;

            float cone_angle_constant = 1.f/256.f;

            bool visualize_cameras = false;
            bool render_with_lens_distortion = false;
            Lens render_lens = {};

            float render_min_transmittance = 0.01f;

            float glow_y_cutoff = 0.f;
            int glow_mode = 0;
        } m_nerf;

        enum EDataType {
            Float,
            Half,
        };

        float m_camera_velocity = 1.0f;
        EColorSpace m_color_space = EColorSpace::Linear;
        ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;

        // 3D stuff
        float m_render_near_distance = 0.0f;
        float m_slice_plane_z = 0.0f;
        bool m_floor_enable = false;
        inline float get_floor_y() const { return m_floor_enable ? m_aabb.min.y()+0.001f : -10000.f; }
        BoundingBox m_raw_aabb;
        BoundingBox m_aabb;
        BoundingBox m_render_aabb;
        Eigen::Matrix3f m_render_aabb_to_local;
        float m_model_translation[3] {};
        float m_model_rotation[3] {};
        float m_model_scale[3] {};

        Eigen::Matrix<float, 3, 4> crop_box(bool nerf_space) const;
        std::vector<Eigen::Vector3f> crop_box_corners(bool nerf_space) const;
        void set_crop_box(Eigen::Matrix<float, 3, 4> m, bool nerf_space);

        // Rendering/UI bookkeeping
        Ema m_training_prep_ms = {EEmaType::Time, 100};
        Ema m_training_ms = {EEmaType::Time, 100};
        Ema m_render_ms = {EEmaType::Time, 100};
        // The frame contains everything, i.e. training + rendering + GUI and buffer swapping
        Ema m_frame_ms = {EEmaType::Time, 100};
        std::chrono::time_point<std::chrono::steady_clock> m_last_frame_time_point;
        std::chrono::time_point<std::chrono::steady_clock> m_last_gui_draw_time_point;
        std::chrono::time_point<std::chrono::steady_clock> m_training_start_time_point;
        Eigen::Array4f m_background_color = {1.0f, 1.0f, 1.0f, 1.0f};

        bool m_vsync = false;

        // Visualization of neuron activations
        int m_visualized_dimension = -1;
        int m_visualized_layer = 0;
        Eigen::Vector2i m_n_views = {1, 1};
        Eigen::Vector2i m_view_size = {1, 1};
        bool m_single_view = true; // Whether a single neuron is visualized, or all in a tiled grid
        float m_picture_in_picture_res = 0.f; // if non zero, requests a small second picture :)

        bool m_imgui_enabled = true; // tab to toggle
        bool m_visualize_unit_cube = false;
        bool m_snap_to_pixel_centers = false;
        bool m_edit_render_aabb = false;

        Eigen::Vector3f m_parallax_shift = {0.0f, 0.0f, 0.0f}; // to shift the viewer's origin by some amount in camera space

        // CUDA stuff
        tcnn::StreamAndEvent m_stream;

        // Hashgrid encoding analysis
        float m_quant_percent = 0.f;
        std::vector<LevelStats> m_level_stats;
        std::vector<LevelStats> m_first_layer_column_stats;
        int m_num_levels = 0;
        int m_histo_level = 0; // collect a histogram for this level
        uint32_t m_base_grid_resolution;
        float m_per_level_scale;
        float m_histo[257] = {};
        float m_histo_scale = 1.f;

        uint32_t m_training_step = 0;
        uint32_t m_training_batch_size = 1 << 18;
        Ema m_loss_scalar = {EEmaType::Time, 100};
        std::vector<float> m_loss_graph = std::vector<float>(256, 0.0f);
        size_t m_loss_graph_samples = 0;

        bool m_train_encoding = true;
        bool m_train_network = true;

        filesystem::path m_data_path;
        filesystem::path m_network_config_path;

        nlohmann::json m_network_config;


        default_rng_t m_rng;

        CudaRenderBuffer m_windowless_render_surface{std::make_shared<CudaSurface2D>()};

        uint32_t network_width(uint32_t layer) const;
        uint32_t network_num_forward_activations() const;

        std::shared_ptr<tcnn::Loss<precision_t>> m_loss;
        // Network & training stuff
        std::shared_ptr<tcnn::Optimizer<precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Encoding<precision_t>> m_encoding;
        std::shared_ptr<tcnn::Network<float, precision_t>> m_network;
        std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> m_trainer;

        std::shared_ptr<NerfNetwork<precision_t>> m_nerf_network;

        struct TrainableEnvmap {
            std::shared_ptr<tcnn::Optimizer<float>> optimizer;
            std::shared_ptr<TrainableBuffer<4, 2, float>> envmap;
            std::shared_ptr<tcnn::Trainer<float, float, float>> trainer;

            Eigen::Vector2i resolution;
            ELossType loss_type;
        } m_envmap;

        struct TrainableDistortionMap {
            std::shared_ptr<tcnn::Optimizer<float>> optimizer;
            std::shared_ptr<TrainableBuffer<2, 2, float>> map;
            std::shared_ptr<tcnn::Trainer<float, float, float>> trainer;
            Eigen::Vector2i resolution;
        } m_distortion;
    };

NGP_NAMESPACE_END
