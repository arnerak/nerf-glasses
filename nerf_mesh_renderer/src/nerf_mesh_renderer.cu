#include "nerf_mesh_renderer.cuh"

#include <imgui/imgui.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui_impl_glfw.h>

#include <glm/gtc/type_ptr.hpp>

#include <fstream>

#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

namespace optix_ptx
{
#include "optix_ptx.h"
}

#include "common.h"
#include "gltf_scene.h"
#include "ngp/testbed.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "orbit_camera.h"
#define FLYTHROUGH_CAMERA_IMPLEMENTATION
#include "flythrough_camera.h"

#include "portable-file-dialogs.h"

#include "floatyremover.h"

__global__ void combineBuffersKernel(
    const float *__restrict__ inDepthBuffer,
    const Eigen::Array4f *__restrict__ inFramebuffer,
    float *outDepthBuffer,
    Eigen::Array4f *outFramebuffer)
{
    const uint32_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;

    float inDepth = inDepthBuffer[pixelIdx];
    if (inDepth < outDepthBuffer[pixelIdx])
    {
        outDepthBuffer[pixelIdx] = inDepth;
        outFramebuffer[pixelIdx] = inFramebuffer[pixelIdx];
    }
}

//__global__ void copyRaytracingBuffersToNerfRays(
//    uint32_t num_elements,
//    const float* __restrict__ inDepthBuffer,
//    const Eigen::Array4f* __restrict__ inFramebuffer,
//    ngp::NerfPayload* payloads
//) {
//    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= num_elements) return;
//
//    auto& payload = payloads[idx];
//    payload.t_surface = inDepthBuffer[idx];
//    payload.surface_color = inFramebuffer[idx];
//}

__global__ void copyRaytracingBuffersToNerfRays(
    uint32_t num_elements,
    const float *__restrict__ inDepthBuffer,
    const Eigen::Array4f *__restrict__ inFramebuffer,
    ngp::NerfPayload *payloads,
    int pitch,
    int mesh_scale)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x = idx % pitch;
    int y = idx / pitch;

    const uint32_t mesh_pitch = pitch * mesh_scale;
    const uint32_t mesh_idx = y * mesh_pitch * mesh_scale + x * mesh_scale;

    if (idx >= num_elements)
        return;

    auto &payload = payloads[idx];

    Eigen::Array4f composite_color{0.f, 0.f, 0.f, 0.f};
    float depth{0.f};

    for (int i = 0; i < mesh_scale; i++)
    {
        for (int j = 0; j < mesh_scale; j++)
        {
            composite_color += inFramebuffer[mesh_idx + i + j * mesh_pitch];
            depth = max(depth, inDepthBuffer[mesh_idx + i + j * mesh_pitch]);
        }
    }
    composite_color /= mesh_scale * mesh_scale;

    payload.t_surface = depth;
    payload.surface_color = composite_color;
}

static void optixContextLogCallback(uint32_t level, const char *tag, const char *message, void * /* cbdata */)
{
    spdlog::debug("[{}][{}]: {}", level, tag, message);
}

static void errorCallback(int error, const char *description)
{
    spdlog::error("GLFW error: {}, code: {}", description, (uint32_t)error);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    const auto renderer = static_cast<NerfMeshRenderer *>(glfwGetWindowUserPointer(window));
    renderer->keyCallback(key, scancode, action, mods);
}

void cursorPosCallback(GLFWwindow *window, double xPos, double yPos)
{
    const auto renderer = static_cast<NerfMeshRenderer *>(glfwGetWindowUserPointer(window));
    renderer->cursorPosCallback(xPos, yPos);
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    const auto renderer = static_cast<NerfMeshRenderer *>(glfwGetWindowUserPointer(window));
    renderer->mouseButtonCallback(button, action, mods);
}

void scrollCallback(GLFWwindow *window, double xOffset, double yOffset)
{
    if (ImGui::GetIO().WantCaptureMouse)
    {
        // Scrolling while mouse is hovering over UI. Only UI should be affected, return early to prevent renderer zoom.
        return;
    }

    const auto renderer = static_cast<NerfMeshRenderer *>(glfwGetWindowUserPointer(window));
    renderer->scrollCallback(xOffset, yOffset);
}

void NerfMeshRenderer::keyCallback(int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
    {
        return;
    }

    if (key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(_window, GLFW_TRUE);
    }
}

void NerfMeshRenderer::cursorPosCallback(double xPos, double yPos)
{
    // const double prevX = _cursorControls.cursorPosX;
    // const double prevY = _cursorControls.cursorPosY;

    // if(_cursorControls.bIsRmbPressed) {
    //     const auto angleX = static_cast<float>(xPos - prevX);  // Horizontal mouse movement -> yaw.
    //     const auto angleY = static_cast<float>(yPos - prevY);  // Vertical mouse movement -> pitch.
    //     _camera->move(angleX, angleY);
    // }

    // _cursorControls.cursorPosX = xPos;
    // _cursorControls.cursorPosY = yPos;
}

void NerfMeshRenderer::mouseButtonCallback(int button, int action, int mods)
{
    if (action == GLFW_PRESS)
	{
		glfwGetCursorPos(_window, &_cursorControls.cursorPosX, &_cursorControls.cursorPosY);
		glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}
	else if (action == GLFW_RELEASE)
	{
		glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
    //if(button == GLFW_MOUSE_BUTTON_RIGHT) {
    //    _cursorControls.bIsRmbPressed = (action == GLFW_PRESS);
    //}
}

void NerfMeshRenderer::scrollCallback(double xOffset, double yOffset)
{
    // Scroll up: yOffset == 1.0 -> but on scroll up want to zoom in, so negate.
    //_camera->zoom(static_cast<float>(-yOffset));
    scroll_offset += yOffset;
}

bool NerfMeshRenderer::handleInput(float deltaTime)
{
	bool forw = glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS;
	bool left = glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS;
	bool back = glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS;
	bool right = glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS;
	int fast = glfwGetKey(_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
	int slow = glfwGetKey(_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
	bool lmb = glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS;
	bool rmb = glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS;


    double xpos, ypos;
    glfwGetCursorPos(_window, &xpos, &ypos);
    //glfwSetCursorPos(_window, _cursorControls.cursorPosX, _cursorControls.cursorPosY);

    if (lmb || rmb || scroll_offset) {
        orbitcam_update(
            cam_pos, cam_pivot, up, cam_look, viewMat,
            deltaTime,
            1.f, 0.2f, xpos - _cursorControls.cursorPosX, ypos - _cursorControls.cursorPosY, scroll_offset, lmb, rmb);

        _cursorControls.cursorPosX = xpos;
        _cursorControls.cursorPosY = ypos;
        updateModelViewProj();
    }

    // flythrough_camera_update(
    // 	cam_pos, cam_look, up, viewMat,
    // 	deltaTime,
    // 	1.f + fast * 3.f - slow * .75f,
    // 	0.2f,
    // 	89.9f,
    // 	xpos - _cursorControls.cursorPosX, ypos - _cursorControls.cursorPosY,
    // 	forw, left, back, right,
    // 	0,
    // 	0,
    // 	0
    // );

    scroll_offset = 0;
        
	return lmb;
}


std::vector<uint8_t> NerfMeshRenderer::dumpDensityGrid()
{
    auto grid_mip_offset = [](uint32_t mip)
    {
        return (128 * 128 * 128) * mip;
    };

    const int grid_size = ngp::NERF_GRIDSIZE();
    const int size = _nerfs.back()->m_nerf.density_grid_bitfield.bytes();

    // allocate memory for density bitfield
    uint8_t *density_grid = new uint8_t[size];
    _nerfs.back()->m_nerf.density_grid_bitfield.copy_to_host(density_grid);

    auto pos_to_idx = ngp::Testbed::Nerf::pos_to_cascaded_grid_idx;

    float grid_size_f = (float)grid_size;

    std::vector<uint8_t> density_grid_mips{};
    density_grid_mips.reserve(size * 8);

    for (int mip = 0; mip < 8; mip++)
    {
        int mip_size = grid_size / (1 << mip);
        int num_cells_set = 0;

        for (int z = 0; z < grid_size; z++)
        {
            float zz = (z / grid_size_f - 0.5f) * (1 << mip) + 0.5f;
            for (int y = 0; y < grid_size; y++)
            {
                float yy = (y / grid_size_f - 0.5f) * (1 << mip) + 0.5f;
                for (int x = 0; x < grid_size; x++)
                {
                    float xx = (x / grid_size_f - 0.5f) * (1 << mip) + 0.5f;
                    Eigen::Vector3f pos{xx, yy, zz};
                    auto idx = pos_to_idx(pos, mip);
                    uint8_t density_byte = density_grid[idx / 8 + grid_mip_offset(mip) / 8];
                    bool is_bit_set = density_byte & (1UL << (idx % 8));
                    num_cells_set += is_bit_set;
                    density_grid_mips.push_back(is_bit_set);
                }
            }
        }
        printf("num cells set for mip %d: %d\n", mip, num_cells_set);
    }

    return density_grid_mips;
}

void NerfMeshRenderer::loadDensityGrid(const std::vector<uint8_t>& density_grid_mips)
{
    const int size = _nerfs.back()->m_nerf.density_grid_bitfield.bytes();
    constexpr int grid_size = ngp::NERF_GRIDSIZE();

    auto grid_mip_offset = [](uint32_t mip)
    {
        return (grid_size * grid_size * grid_size) * mip;
    };

    // allocate memory for density bitfield
    uint8_t *density_grid = new uint8_t[size]{ 0 };

    auto pos_to_idx = ngp::Testbed::Nerf::pos_to_cascaded_grid_idx;

    float grid_size_f = (float)grid_size;


    for (int mip = 0; mip < 8; mip++)
    {
        int mip_size = grid_size / (1 << mip);
        int num_cells_set = 0;

        for (int z = 0; z < grid_size; z++)
        {
            float zz = (z / grid_size_f - 0.5f) * (1 << mip) + 0.5f;
            for (int y = 0; y < grid_size; y++)
            {
                float yy = (y / grid_size_f - 0.5f) * (1 << mip) + 0.5f;
                for (int x = 0; x < grid_size; x++)
                {
                    float xx = (x / grid_size_f - 0.5f) * (1 << mip) + 0.5f;
                    Eigen::Vector3f pos{xx, yy, zz};
                    auto idx = pos_to_idx(pos, mip);
                    uint8_t density_byte = density_grid[idx / 8 + grid_mip_offset(mip) / 8];
                    bool is_bit_set = density_grid_mips.at(x + y * 128 + z * 128 * 128 + mip * 128 * 128 * 128);
                    num_cells_set += is_bit_set;
                    density_byte = density_byte | (is_bit_set << (idx % 8));
                    density_grid[idx / 8 + grid_mip_offset(mip) / 8] = density_byte;
                }
            }
        }

        printf("num cells set for mip %d: %d\n", mip, num_cells_set);
    }

    _nerfs.back()->m_nerf.density_grid_bitfield.copy_from_host(density_grid);
}

void NerfMeshRenderer::dumpDensityGrid(const std::string& filename)
{
    auto density_grid_mips = dumpDensityGrid();
    std::ofstream file;
    file.open(filename, std::ios::binary);
    file.write((char*)density_grid_mips.data(), density_grid_mips.size());
    file.close();
}

void NerfMeshRenderer::loadDensityGrid(const std::string& filename)
{
    const int size = _nerfs.back()->m_nerf.density_grid_bitfield.bytes();

    std::vector<uint8_t> density_grid_mips(size * 8);
    std::ifstream file;
    file.open(filename, std::ios::binary);
    file.read((char*)density_grid_mips.data(), size*8);
    file.close();

    loadDensityGrid(density_grid_mips);
}

std::ostream& operator<<(std::ostream& os, const glm::vec3& vector) {
    os << "(" << vector.x << ", " << vector.y << ", " << vector.z << ")";
    return os;
}

NerfMeshRenderer::NerfMeshRenderer(int width, int height) : SCREEN_WIDTH(width), SCREEN_HEIGHT(height)
{
    render_width = SCREEN_WIDTH * render_size_factor;
    render_height = SCREEN_HEIGHT * render_size_factor;

    flythrough_camera_update(cam_pos, cam_look, up, viewMat,
                             0.016f, 0, 0, 90.f, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    updateModelViewProj();

    // _meshes.emplace_back(GltfLoader::load("../assets/meshes/Suzanne/glTF/Suzanne.gltf"));
    // _meshes.back()->nodes[0]->scale = {0.2f, 0.2f, 0.2f};


    // == GLFW =========================================================================================================
    glfwSetErrorCallback(errorCallback);

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    _window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "NeRF-Mesh-Renderer", nullptr, nullptr);
    if (_window == nullptr)
    {
        spdlog::error("Window creation failed.");
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // Allows us to retrieve a pointer to NerfMeshRenderer from _window.
    glfwSetWindowUserPointer(_window, this);

    glfwSetKeyCallback(_window, ::keyCallback);
    glfwSetCursorPosCallback(_window, ::cursorPosCallback);
    glfwSetMouseButtonCallback(_window, ::mouseButtonCallback);
    glfwSetScrollCallback(_window, ::scrollCallback);

    glfwMakeContextCurrent(_window);

    glfwSwapInterval(1);

    // == GLAD =========================================================================================================

    gladLoadGL(glfwGetProcAddress);

    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    lastTime = glfwGetTime();
    lastFpsTime = lastTime;

    // == ImGui ========================================================================================================

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    //    ImGui::StyleColorsLight();
    //    ImGui::StyleColorsClassic();
    ImGui::StyleColorsDark();

    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.2f;

    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    // == OptiX ========================================================================================================

    _cudaOutputBuffer = new CudaOutputBuffer<uchar4>{ CudaOutputBufferType::ZERO_COPY, render_width * mesh_render_size_factor, render_height * mesh_render_size_factor };

    CUDA_CHECK(cudaStreamCreate(&_cuStream));

    initOptix();

    // == Instant-NGP ==================================================================================================

    _ngpRenderTexture = std::make_shared<ngp::GLTexture>();
    _ngpRenderTexture->resize({render_width, render_height}, 4, false);
    _ngpRenderTexture->surface();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_dDepthBuffer), render_width * render_height * sizeof(float) * mesh_render_size_factor* mesh_render_size_factor));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_dFramebuffer), render_width * render_height * sizeof(Eigen::Array4f) * mesh_render_size_factor* mesh_render_size_factor));


    // =================================================================================================================

    _bIsInitialized = true;
}

void NerfMeshRenderer::cleanup()
{
    if (!_bIsInitialized)
    {
        spdlog::error("NerfMeshRenderer::cleanup() called with incomplete initialization.");
        return;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_dFramebuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_dDepthBuffer)));

    cleanupOptix();
    cleanupMeshes();
    cleanupNerfs();

    CUDA_CHECK(cudaStreamDestroy(_cuStream));

    delete _cudaOutputBuffer;
    // delete _camera;

    GL_CHECK(glDeleteProgram(_glProgram));

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(_window);
    glfwTerminate();
}

static glm::vec3 rotVec = { 0.f, -0.5f, 0.f };
void NerfMeshRenderer::update() {


    // for (auto mesh : _meshes) {
    //     mesh->nodes[0]->RotateAroundAxis({ 1.f, 0.f, 0.f},  rotVec, 0.5f);
    //         createOptixIas();
    // }
}

void NerfMeshRenderer::run()
{
    while (frame()) { }
}

bool NerfMeshRenderer::frame()
{
    int framebufferWidth;
    int frameBufferHeight;

    // While glfwWaitEvents() is the more logical choice for this app,
    // we are interested in seeing the performance (~FPS), so use glfwPollEvents() instead.
    //        glfwWaitEvents();
    glfwPollEvents();

    glfwGetFramebufferSize(_window, &framebufferWidth, &frameBufferHeight);

    GL_CHECK(glViewport(0, 0, framebufferWidth, frameBufferHeight));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    update();
    render_frame();

    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(_window);

    const double currentTime = glfwGetTime();
    const auto deltaTime = currentTime - lastTime;
    if (!ImGui::GetIO().WantCaptureMouse)
        handleInput(deltaTime);
    lastTime = currentTime;
    numberFrames++;
    if (currentTime - lastFpsTime >= 1.0)
    {
        _imGuiOptions.fps = static_cast<double>(numberFrames);
        numberFrames = 0;
        lastFpsTime = currentTime;
    }
    
    return !glfwWindowShouldClose(_window);
}

void NerfMeshRenderer::render_frame()
{
    // == ImGui ========================================================================================================

    gui();

    // == Content ======================================================================================================

    //_nerfs[0]->m_nerf.tracer.rays_init().payload;

    uint32_t numPixels = render_width * render_height;
    if (_nerfs.size() > 0 && _nerfs[0]->m_nerf.tracer.rays_init().payload)
    {
        launchOptix();
        tcnn::linear_kernel(copyRaytracingBuffersToNerfRays, 0, 0, numPixels, _dDepthBuffer, _dFramebuffer, _nerfs[0]->m_nerf.tracer.rays_init().payload, render_width, (int)mesh_render_size_factor);
    }

    if (!_nerfs.empty())
    {
        GL_CHECK(glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD));
        GL_CHECK(glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

        for (const auto &nerf : _nerfs)
        {
            nerf->m_camera << viewProjectionMat;
            // if ((nerf->m_smoothed_camera - nerf->m_camera).norm() >= 0.001f) {
            //  Reset accumulation in case camera position has changed significantly since last frame.
            //nerf->reset_accumulation(true, true);
            //}

            nerf->m_smoothed_camera = nerf->m_camera;

            // Clear needed? Seems to work without...
            // nerf->m_render_surfaces.front().clear_frame(0);
            nerf->render_frame(nerf->m_smoothed_camera, nerf->m_smoothed_camera, Eigen::Vector4f::Zero(), nerf->m_render_surfaces.front());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // We have at least one NeRF loaded. Copy the first one's depth and framebuffer over.
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(_dDepthBuffer), _nerfs[0]->m_render_surfaces.front().depth_buffer(), numPixels * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(_dFramebuffer), _nerfs[0]->m_render_surfaces.front().frame_buffer(), numPixels * sizeof(Eigen::Array4f), cudaMemcpyDeviceToDevice));

        // If there are more NeRFs, combine their depth and framebuffers.
        for (int i = 1; i < _nerfs.size(); ++i)
        {
            const auto &nerf = _nerfs[i];

            const uint32_t numBlocks = (numPixels + 127) / 128;
            combineBuffersKernel<<<numBlocks, 128>>>(
                nerf->m_render_surfaces.front().depth_buffer(),
                nerf->m_render_surfaces.front().frame_buffer(),
                _dDepthBuffer,
                _dFramebuffer);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

void NerfMeshRenderer::gui()
{
    static std::string last_dir = pfd::path::home();
    static bool trajectory_running{false};
    static uint8_t *frame_buf = new uint8_t[SCREEN_WIDTH * SCREEN_HEIGHT * 3];
    static float trajectory_cam_angle = 0.5;
    static int trajectory_idx = 0;
    static float trajectory_distance = 1.1f;
    static float trajectory_height = 0.1f;
    static float trajectory_start_angle = 0.5f;
    static float trajectory_end_angle = 2.5f;
    static int trajectory_num_images = 10;
    static glm::vec3 trajectory_lookat{0.f, 0.f, 0.f};

    if (_nerfs.size() > 0)
    {
        ImDrawList *list = ImGui::GetBackgroundDrawList();
        // list->AddCallback([](const ImDrawList*, const ImDrawCmd*) {
        //	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        //	glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        // }, nullptr);
        _nerfs[0]->m_render_textures.front()->blit_from_cuda_mapping();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        list->AddImageQuad((ImTextureID)(size_t)_nerfs[0]->m_render_textures.front()->texture(), ImVec2{0.f, 0.f}, ImVec2{(float)SCREEN_WIDTH, 0.f}, ImVec2{(float)SCREEN_WIDTH, (float)SCREEN_HEIGHT}, ImVec2{0.f, (float)SCREEN_HEIGHT}, ImVec2(0, 1), ImVec2(1, 1), ImVec2(1, 0), ImVec2(0, 0));
    }
    else
    {
    }

    if (trajectory_running)
    {
        if (trajectory_idx > 0)
        {
            glReadBuffer(GL_FRONT);
            glReadPixels(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, frame_buf);
            stbi_flip_vertically_on_write(1);
            stbi_write_jpg(("trajectory_" + std::to_string(trajectory_idx) + ".jpg").c_str(), SCREEN_WIDTH, SCREEN_HEIGHT, 3, frame_buf, 95);
            std::ofstream transform_file;
            transform_file.open("transform_" + std::to_string(trajectory_idx));
            Eigen::IOFormat json_format(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");
            transform_file << viewProjectionMat.format(json_format);
            transform_file.close();
        }
        trajectory_cam_angle += (trajectory_end_angle - trajectory_start_angle) / trajectory_num_images;
        trajectory_idx++;
        if (trajectory_cam_angle >= trajectory_end_angle)
            trajectory_running = false;

        cam_pos[0] = cosf(trajectory_cam_angle) * trajectory_distance;
        cam_pos[1] = trajectory_height;
        cam_pos[2] = sinf(trajectory_cam_angle) * trajectory_distance;
        glm::vec3 look{trajectory_lookat.x - cam_pos[0], trajectory_lookat.y - cam_pos[1], trajectory_lookat.z - cam_pos[2]};
        look = glm::normalize(look);
        cam_look[0] = look.x;
        cam_look[1] = look.y;
        cam_look[2] = look.z;
        flythrough_camera_look_to(cam_pos, cam_look, up, viewMat, 0);
        updateModelViewProj();
        return;
    }

    ImVec2 windowSize{SCREEN_WIDTH / 4.0f, SCREEN_HEIGHT};
    ImVec2 windowPos{0.0f, 0.0f};
    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);

    ImGui::Begin("NeRF-Mesh-Renderer");

    if (ImGui::CollapsingHeader("NeRF"))
    {
        if (ImGui::Button("Load NeRF"))
        {
            //loadNerf(loadNerfInputText);
            auto selectedFiles = pfd::open_file("Choose a msgpack file", last_dir).result();
            if (selectedFiles.size() == 1) {
                last_dir = selectedFiles.at(0);
                loadNerf(selectedFiles.at(0));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear NeRFs"))
        {
            cleanupNerfs();
        }
        if (ImGui::Button("Load Density Grid"))
        {
            //loadNerf(loadNerfInputText);
            auto selectedFiles = pfd::open_file("Choose a density grid file", last_dir).result();
            if (selectedFiles.size() == 1) {
                last_dir = selectedFiles.at(0);
                loadDensityGrid(selectedFiles.at(0));
            }
        }
        if (ImGui::Button("Dump Density Grid"))
        {
            //loadNerf(loadNerfInputText);
            auto selectedFile = pfd::save_file("Save a density grid file", last_dir).result();
            if (selectedFile.size() > 0) {
                last_dir = selectedFile;
                dumpDensityGrid(selectedFile);
            }
        }


        for (int i = 0; i < _nerfs.size(); ++i)
        {
            const auto nerf = _nerfs[i];
            if (nerf == nullptr)
            {
                continue;
            }

            ImGui::Separator();

            if (ImGui::TreeNode((nerf->m_name + "_" + std::to_string(i)).c_str()))
            {
                ImGui::DragFloat3("Position", nerf->m_model_translation, 0.05f);
                ImGui::DragFloat3("Rotation", nerf->m_model_rotation, 0.01f);
                ImGui::TreePop();
                static float rot = 0.f;
                // rot += 0.05f;
                // nerf->modelMat.block<3,3>(0,0) << cosf(rot), 0, sinf(rot), 0, 1, 0, -sinf(rot), 0, cosf(rot);

                nerf->m_render_aabb.translate();
            }
        }
    }

    if (ImGui::CollapsingHeader("Mesh"))
    {
        if (ImGui::Button("Load mesh"))
        {
            //loadMesh(gltfModelPaths[selectedGltfModelIndex]);
            auto selectedFiles = pfd::open_file("Choose a gltf file", last_dir).result();
            if (selectedFiles.size() == 1) {
                last_dir = selectedFiles.at(0);
                loadMesh(selectedFiles.at(0));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear meshes"))
        {
            cleanupMeshes();
            cleanupOptixMeshGas();
            cleanupOptixIas();

            // Rebuild SBT and IAS to reflect new state.
            createOptixSbt();
            createOptixIas();
        }

        bool bRebuildIas = false;
        for (int i = 0; i < _meshes.size(); ++i)
        {
            const GltfScene *mesh = _meshes[i];
            if (mesh == nullptr || mesh->nodes.empty())
            {
                continue;
            }

            ImGui::Separator();

            if (ImGui::TreeNode((mesh->getName() + "_" + std::to_string(i)).c_str()))
            {
                GltfNode *node = mesh->nodes[0];
                bRebuildIas = bRebuildIas || ImGui::DragFloat3("Position", glm::value_ptr(node->translation), 0.05f);
                bRebuildIas = bRebuildIas || ImGui::DragFloat3("Scale", glm::value_ptr(node->scale), 0.05f);
                ImGui::TreePop();

                static glm::vec3 moveDirection { 0.f, -1.f, 0.f };
                static bool colliding = false;
                ImGui::DragFloat3("move dir", glm::value_ptr(moveDirection));
                if (ImGui::Button("collide")) {
                    colliding = true;
                }
                if (colliding) {
                    colliding = !collide(moveDirection, *node);
                    cleanupOptix();
                    initOptix();
                }
            }
        }

        if (bRebuildIas)
        {
            createOptixIas();
        }
    }

    if (ImGui::CollapsingHeader("Light"))
    {
        ImGui::DragFloat3("Pos", lightPos, 0.05f);
    }

    if (ImGui::CollapsingHeader("Camera Trajectory"))
    {
        bool hasChangedParams = false, hasChangedStart = false, hasChangedEnd = false;
        hasChangedParams |= ImGui::DragFloat("Camera Height", &trajectory_height, 0.01f);
        hasChangedParams |= ImGui::DragFloat("Camera Distance", &trajectory_distance, 0.01f);
        hasChangedParams |= ImGui::DragFloat3("Look At", &trajectory_lookat.x, 0.01f);
        if (hasChangedStart = ImGui::DragFloat("Start angle", &trajectory_start_angle, 0.01f, 0.f, 6.28f))
            trajectory_cam_angle = trajectory_start_angle;
        if (hasChangedEnd = ImGui::DragFloat("End angle", &trajectory_end_angle, 0.01f, 0, 6.28f))
            trajectory_cam_angle = trajectory_end_angle;
        
        ImGui::DragInt("Num images", &trajectory_num_images);

        if (hasChangedParams || hasChangedStart || hasChangedEnd)
        {
            cam_pos[0] = cosf(trajectory_cam_angle) * trajectory_distance;
            cam_pos[1] = trajectory_height;
            cam_pos[2] = sinf(trajectory_cam_angle) * trajectory_distance;
            glm::vec3 look{trajectory_lookat.x - cam_pos[0], trajectory_lookat.y - cam_pos[1], trajectory_lookat.z - cam_pos[2]};
            look = glm::normalize(look);
            cam_look[0] = look.x;
            cam_look[1] = look.y;
            cam_look[2] = look.z;
            flythrough_camera_look_to(cam_pos, cam_look, up, viewMat, 0);
            updateModelViewProj();
        }
        if (ImGui::Button("Run Trajectory"))
        {
            trajectory_running = true;
            trajectory_cam_angle = trajectory_start_angle;
            trajectory_idx = 0;
        }
    }

    if (ImGui::CollapsingHeader("Statistics"))
    {
        // Frame time and FPS.
        {
            char fpsText[32];
            sprintf(fpsText, "FPS: %.2f", _imGuiOptions.fps);

            char frameTimeText[32];
            sprintf(frameTimeText, "Frame time: %.2f ms", (1000.0 / _imGuiOptions.fps));

            if (_imGuiOptions.fps < 24.0)
            {
                constexpr ImVec4 colorRed(0.9f, 0.0f, 0.0f, 1.0f);
                ImGui::TextColored(colorRed, fpsText);
                ImGui::TextColored(colorRed, frameTimeText);
            }
            else
            {
                ImGui::Text(fpsText);
                ImGui::Text(frameTimeText);
            }
        }

        // GPU VRAM statistics.
        {
            cudaDeviceProp deviceProps{};
            cudaGetDeviceProperties(&deviceProps, 0);

            size_t freeMemory;
            size_t totalMemory;
            cudaMemGetInfo(&freeMemory, &totalMemory);

            constexpr auto gb = static_cast<float>(1 << 30);
            const float freeInGb = static_cast<float>(freeMemory) / gb;
            const float totalInGb = static_cast<float>(totalMemory) / gb;
            const float usedInGb = totalInGb - freeInGb;

            char progressText[32];
            sprintf(progressText, "%.2f / %.2f GB", usedInGb, totalInGb);

            ImVec2 progressRange(0.0f, 0.0f);
            ImGui::ProgressBar(usedInGb / totalInGb, progressRange, progressText);
            ImGui::SameLine();
            ImGui::Text(deviceProps.name);
        }
    }

    // if (ImGui::Button("Remove Floaties")) {
    //     removeFloaties();
    // }

#if !defined(NDEBUG)
    if (ImGui::CollapsingHeader("Debug"))
    {
        ImGui::Checkbox("Show ImGui's demo window", &_imGuiOptions.bShowDemoWindow);
    }

    if (_imGuiOptions.bShowDemoWindow)
    {
        ImGui::ShowDemoWindow();
    }
#endif

    ImGui::End();
}


void NerfMeshRenderer::orbit(float delta_azimuth, float delta_polar, float delta_zoom) {
    orbitcam(cam_pos, cam_pivot, up, cam_look, viewMat, delta_polar, delta_azimuth, delta_zoom);
    updateModelViewProj();
}

void NerfMeshRenderer::removeFloaties() {
    auto density_grid_mips = dumpDensityGrid();
    auto start = std::chrono::high_resolution_clock::now();
    NgpGrid grid{ density_grid_mips };
    auto clusters = grid.cluster();
    const auto& largestCluster = *std::max_element(clusters.begin(), clusters.end(),
        [](const NgpGrid::DensityPointSet& a, const NgpGrid::DensityPointSet& b) {
            return NgpGrid::point_set_importance(a) < NgpGrid::point_set_importance(b);
        });
    NgpGrid::to_ngp_grid(density_grid_mips.data(), largestCluster);
    loadDensityGrid(density_grid_mips);
    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << clusters.size() << "   " << duration.count() << " ms" << std::endl;
}

void NerfMeshRenderer::updateModelViewProj()
{
    float aspectRatio = SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    float vLength = tanf(0.5f * 45);
    float uLength = vLength * aspectRatio;
    glm::vec3 u{viewMat[0], viewMat[4], viewMat[8]};
    glm::vec3 v{viewMat[1], viewMat[5], viewMat[9]};
    glm::vec3 w{-viewMat[2], -viewMat[6], -viewMat[10]};
    u = u * uLength;
    v = v * vLength;
    viewProjectionMat << u.x, v.x, -viewMat[2], cam_pos[0],
        u.y, v.y, -viewMat[6], cam_pos[1],
        u.z, v.z, -viewMat[10], cam_pos[2];

    
    for (const auto &nerf : _nerfs)
    {
        nerf->m_camera << viewProjectionMat;
        nerf->reset_accumulation(true);
    }
}

GltfScene* NerfMeshRenderer::loadMesh(
    const std::string &filename, 
    const Eigen::Vector3f& t,
    const Eigen::Vector3f& s,
    const Eigen::Vector4f& r
    )
{
    try
    {
        GltfScene* mesh{ GltfLoader::load(filename) };
        _meshes.emplace_back(mesh);
        mesh->nodes[0]->translation = { t.x(), t.y(), t.z() };
        mesh->nodes[0]->scale = { s.x(), s.y(), s.z() };
        mesh->nodes[0]->rotation = { r.x(), r.y(), r.z(), r.w() };
        // Inefficient: build everything from scratch again (better: e.g. work with a linear allocator).
        cleanupOptix();
        initOptix();
        return mesh;
    }
    catch (std::exception &e)
    {
        spdlog::error("Failed to load mesh {}: {}", filename, e.what());
        return nullptr;
    }
}

ngp::Testbed* NerfMeshRenderer::loadNerf(const std::string &filename)
{
    try
    {
        // Extract name from file path, from https://stackoverflow.com/a/24386991/20601665.
        const std::string nameWithExtension = filename.substr(filename.find_last_of("/\\") + 1);
        const std::string::size_type lastDotIndex{nameWithExtension.find_last_of('.')};
        const std::string name = nameWithExtension.substr(0, lastDotIndex);

        ngp::Testbed* nerf { new ngp::Testbed(name) };
        nerf->load_snapshot(filename);
        nerf->m_window_res = {SCREEN_WIDTH, SCREEN_HEIGHT};
        // nerf->set_fov(_camera->fovY());
        nerf->set_fov(45.f);
        if (_nerfs.empty())
        {
            nerf->m_render_textures = {_ngpRenderTexture};
        }
        else
        {
            nerf->m_render_textures = _nerfs[0]->m_render_textures;
        }
        nerf->m_render_surfaces.emplace_back(nerf->m_render_textures.front());
        nerf->m_render_surfaces.front().resize({render_width, render_height});
        nerf->m_camera << viewProjectionMat;
        _nerfs.emplace_back(nerf);
        return nerf;
    }
    catch (std::exception &e)
    {
        spdlog::error("Failed to load NeRF {}: {}", filename, e.what());
        return nullptr;
    }
}

void NerfMeshRenderer::initOptix()
{
    OptixPipelineCompileOptions pipelineCompileOptions{};

    createOptixContext();
    createOptixModule(pipelineCompileOptions);
    createOptixProgramGroups();
    createOptixPipeline(pipelineCompileOptions);
    createOptixSbt();
    for (const auto mesh : _meshes)
    {
        createOptixMeshGas(mesh);
    }
    createOptixIas();
}

void NerfMeshRenderer::createOptixContext()
{
    CUDA_CHECK(cudaFree(nullptr)); // Initializes CUDA.

    OPTIX_CHECK(optixInit());

    CUcontext cuContext = nullptr; // nullptr -> OptiX will use the current CUDA context.
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = &optixContextLogCallback;
    options.logCallbackLevel = 4; // 4: Status/progress messages.
#if !defined(NDEBUG)
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &_optixContext));
}

void NerfMeshRenderer::createOptixModule(OptixPipelineCompileOptions &pipelineCompileOptions)
{
    OptixModuleCompileOptions moduleCompileOptions{};
#if !defined(NDEBUG)
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimization.
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 4;
    pipelineCompileOptions.numAttributeValues = 3;                     // TODO: same value as numPayloads? What exactly is this?
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // Better: should be STACK_OVERFLOW?
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    size_t ptxInputSize = sizeof(optix_ptx::optix_scene_ptx);
    const auto ptxInput = reinterpret_cast<const char *>(optix_ptx::optix_scene_ptx);

    char optixLog[2048];
    size_t logSize = sizeof(optixLog);

    OPTIX_CHECK(optixModuleCreateFromPTX(
        _optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxInput,
        ptxInputSize,
        optixLog,
        &logSize,
        &_optixModule));
}

void NerfMeshRenderer::createOptixProgramGroups()
{
    char optixLog[2048];
    size_t logSize;
    OptixProgramGroupOptions programGroupOptions{};

    OptixProgramGroupDesc raygenProgramGroupDesc{};
    raygenProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenProgramGroupDesc.raygen.module = _optixModule;
    raygenProgramGroupDesc.raygen.entryFunctionName = "__raygen__rg";
    logSize = sizeof(optixLog);
    OPTIX_CHECK(optixProgramGroupCreate(
        _optixContext,
        &raygenProgramGroupDesc,
        1,
        &programGroupOptions,
        optixLog,
        &logSize,
        &_optixRaygenProgramGroup));

    OptixProgramGroupDesc missProgramGroupDesc{};
    missProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missProgramGroupDesc.miss.module = _optixModule;
    missProgramGroupDesc.miss.entryFunctionName = "__miss__ms";
    logSize = sizeof(optixLog);
    OPTIX_CHECK(optixProgramGroupCreate(
        _optixContext,
        &missProgramGroupDesc,
        1,
        &programGroupOptions,
        optixLog,
        &logSize,
        &_optixMissProgramGroup));

    OptixProgramGroupDesc hitgroupMeshProgramGroupDesc{};
    hitgroupMeshProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupMeshProgramGroupDesc.hitgroup.moduleCH = _optixModule;
    hitgroupMeshProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    logSize = sizeof(optixLog);
    OPTIX_CHECK(optixProgramGroupCreate(
        _optixContext,
        &hitgroupMeshProgramGroupDesc,
        1,
        &programGroupOptions,
        optixLog,
        &logSize,
        &_optixHitgroupMeshProgramGroup));
}

void NerfMeshRenderer::createOptixPipeline(OptixPipelineCompileOptions &pipelineCompileOptions)
{
    const uint32_t maxTraceDepth = 1;
    OptixProgramGroup programGroups[] = {
        _optixRaygenProgramGroup,
        _optixMissProgramGroup,
        _optixHitgroupMeshProgramGroup,
    };

    OptixPipelineLinkOptions pipelineLinkOptions{};
    pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char optixLog[2048];
    size_t logSize = sizeof(optixLog);

    OPTIX_CHECK(optixPipelineCreate(
        _optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
        optixLog,
        &logSize,
        &_optixPipeline));

    OptixStackSizes stackSizes{};
    for (const auto &programGroup : programGroups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroup, &stackSizes));
    }

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t directCallableStackSize;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        maxTraceDepth,
        0, // maxCCDepth
        0, // maxDCDepth
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &directCallableStackSize));
    OPTIX_CHECK(optixPipelineSetStackSize(
        _optixPipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        directCallableStackSize,
        2 // maxTraversableDepth (IAS -> GAS)
        ));
}

void NerfMeshRenderer::createOptixSbt()
{
    // Ray generation.
    CUdeviceptr dRaygenRecord;
    const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dRaygenRecord), raygenRecordSize));

    RayGenSbtRecord rayGenSbtRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(_optixRaygenProgramGroup, &rayGenSbtRecord));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(dRaygenRecord),
        &rayGenSbtRecord,
        raygenRecordSize,
        cudaMemcpyHostToDevice));

    // Ray miss.
    CUdeviceptr dMissRecord;
    size_t missRecordSize = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dMissRecord), missRecordSize));

    MissSbtRecord missSbtRecord{};
    missSbtRecord.data.backgroundColor = {0.0f, 0.0f, 0.0f};
    OPTIX_CHECK(optixSbtRecordPackHeader(_optixMissProgramGroup, &missSbtRecord));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(dMissRecord),
        &missSbtRecord,
        missRecordSize,
        cudaMemcpyHostToDevice));

    // Ray hit.
    std::vector<HitGroupSbtRecord> hitGroupSbtRecords;

    for (const auto mesh : _meshes)
    {
        for (const auto meshPrimitive : mesh->getMeshPrimitives())
        {
            std::vector<uint16_t> indices = meshPrimitive->indices;
            const size_t indicesSizeInBytes = indices.size() * sizeof(uint16_t);
            CUdeviceptr dIndices = 0u;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dIndices), indicesSizeInBytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(dIndices), indices.data(), indicesSizeInBytes, cudaMemcpyHostToDevice));

            std::vector<glm::vec3> normals = meshPrimitive->normals;
            const size_t normalsSizeInBytes = normals.size() * sizeof(glm::vec3);
            CUdeviceptr dNormals = 0u;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dNormals), normalsSizeInBytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(dNormals), normals.data(), normalsSizeInBytes, cudaMemcpyHostToDevice));

            std::vector<glm::vec4> tangents = meshPrimitive->tangents;
            const size_t tangentsSizeInBytes = tangents.size() * sizeof(glm::vec4);
            CUdeviceptr dTangents = 0u;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTangents), tangentsSizeInBytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(dTangents), tangents.data(), tangentsSizeInBytes, cudaMemcpyHostToDevice));

            std::vector<glm::vec2> texCoords = meshPrimitive->texCoords;
            const size_t texCoordsSizeInBytes = texCoords.size() * sizeof(glm::vec2);
            CUdeviceptr dTexCoords = 0u;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTexCoords), texCoordsSizeInBytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(dTexCoords), texCoords.data(), texCoordsSizeInBytes, cudaMemcpyHostToDevice));

            const auto material = meshPrimitive->material;

            HitGroupSbtRecord hitGroupSbtRecord{};
            OPTIX_CHECK(optixSbtRecordPackHeader(_optixHitgroupMeshProgramGroup, &hitGroupSbtRecord));
            hitGroupSbtRecord.data.geometryData.mesh.dIndices = reinterpret_cast<uint16_t *>(dIndices);
            hitGroupSbtRecord.data.geometryData.mesh.dNormals = reinterpret_cast<glm::vec3 *>(dNormals);
            hitGroupSbtRecord.data.geometryData.mesh.dTangents = reinterpret_cast<glm::vec4 *>(dTangents);
            hitGroupSbtRecord.data.geometryData.mesh.dTexCoords = reinterpret_cast<glm::vec2 *>(dTexCoords);
            hitGroupSbtRecord.data.geometryData.mesh.emissiveFactor = material->emissiveFactor;
            if (material->emissiveTexture != nullptr)
            {
                hitGroupSbtRecord.data.geometryData.mesh.emissiveTexture = material->emissiveTexture->getCudaTextureObject();
            }
            hitGroupSbtRecord.data.geometryData.mesh.baseColorFactor = material->baseColor;
            if (material->baseColorTexture != nullptr)
            {
                hitGroupSbtRecord.data.geometryData.mesh.baseColorTexture = material->baseColorTexture->getCudaTextureObject();
            }
            hitGroupSbtRecord.data.geometryData.mesh.metallicFactor = material->metallicFactor;
            hitGroupSbtRecord.data.geometryData.mesh.roughnessFactor = material->roughnessFactor;
            if (material->metallicRoughnessTexture != nullptr)
            {
                hitGroupSbtRecord.data.geometryData.mesh.metallicRoughnessTexture = material->metallicRoughnessTexture->getCudaTextureObject();
            }
            hitGroupSbtRecord.data.geometryData.mesh.normalScale = material->normalScale;
            if (material->normalTexture != nullptr)
            {
                hitGroupSbtRecord.data.geometryData.mesh.normalTexture = material->normalTexture->getCudaTextureObject();
            }
            hitGroupSbtRecord.data.geometryData.mesh.occlusionStrength = material->occlusionStrength;
            if (material->occlusionTexture != nullptr)
            {
                hitGroupSbtRecord.data.geometryData.mesh.occlusionTexture = material->occlusionTexture->getCudaTextureObject();
            }
            hitGroupSbtRecords.emplace_back(hitGroupSbtRecord);
        }
    }

    CUdeviceptr dHitgroupRecord;
    size_t hitgroupRecordSize = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dHitgroupRecord), hitgroupRecordSize * hitGroupSbtRecords.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(dHitgroupRecord),
        hitGroupSbtRecords.data(),
        hitGroupSbtRecords.size() * hitgroupRecordSize,
        cudaMemcpyHostToDevice));

    // Combine into SBT.
    _optixSbt.raygenRecord = dRaygenRecord;
    _optixSbt.missRecordBase = dMissRecord;
    _optixSbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    _optixSbt.missRecordCount = 1;
    _optixSbt.hitgroupRecordBase = dHitgroupRecord;
    _optixSbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    _optixSbt.hitgroupRecordCount = hitGroupSbtRecords.size();
}

void NerfMeshRenderer::createOptixMeshGas(const GltfScene *mesh)
{
    MeshGas meshGas{};

    OptixAccelBuildOptions accelBuildOptions{};
    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    uint32_t triangleInputFlags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

    // Create build inputs per mesh primitive.
    std::vector<OptixBuildInput> buildInputs;
    for (const auto meshPrimitive : mesh->getMeshPrimitives())
    {
        // Upload vertices and (if present) indices onto GPU buffers.
        const std::vector<glm::vec3> vertices = meshPrimitive->positions;
        const size_t verticesSizeInBytes = vertices.size() * sizeof(glm::vec3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&meshGas.dVertexBuffer), verticesSizeInBytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(meshGas.dVertexBuffer), vertices.data(), verticesSizeInBytes, cudaMemcpyHostToDevice));

        // Use them to construct the acceleration structure.
        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexBuffers = &meshGas.dVertexBuffer;
        buildInput.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = 0u; // Tightly packed.
        buildInput.triangleArray.flags = triangleInputFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        if (!meshPrimitive->indices.empty())
        {
            std::vector<uint16_t> indices = meshPrimitive->indices;
            const size_t indicesSizeInBytes = indices.size() * sizeof(uint16_t);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&meshGas.dIndexBuffer), indicesSizeInBytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(meshGas.dIndexBuffer), indices.data(), indicesSizeInBytes, cudaMemcpyHostToDevice));

            buildInput.triangleArray.indexBuffer = meshGas.dIndexBuffer;
            buildInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size() / 3);
            buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
            buildInput.triangleArray.indexStrideInBytes = 0u; // Tightly packed.
        }

        buildInputs.emplace_back(buildInput);
    }

    // Combine all mesh primitives into one GAS for the entire mesh.
    OptixAccelBufferSizes gasBufferSizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        _optixContext,
        &accelBuildOptions,
        buildInputs.data(),
        buildInputs.size(),
        &gasBufferSizes));

    CUdeviceptr dTemp = 0u;
    CUdeviceptr dTempOutput = 0u;
    CUdeviceptr dTempCompactedSizes = 0u;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTemp), gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTempOutput), gasBufferSizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTempCompactedSizes), sizeof(size_t)));

    OptixAccelEmitDesc accelEmitDesc{};
    accelEmitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    accelEmitDesc.result = dTempCompactedSizes;

    OPTIX_CHECK(optixAccelBuild(
        _optixContext,
        nullptr,
        &accelBuildOptions,
        buildInputs.data(),
        buildInputs.size(),
        dTemp,
        gasBufferSizes.tempSizeInBytes,
        dTempOutput,
        gasBufferSizes.outputSizeInBytes,
        &meshGas.handle,
        &accelEmitDesc,
        1));

    // Compaction.
    size_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(&compactedSize), reinterpret_cast<void *>(dTempCompactedSizes), sizeof(size_t), cudaMemcpyDeviceToHost));
    if (gasBufferSizes.outputSizeInBytes > compactedSize)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&meshGas.dBuffer), compactedSize));
        OPTIX_CHECK(optixAccelCompact(_optixContext, nullptr, meshGas.handle, meshGas.dBuffer, compactedSize, &meshGas.handle));
    }
    else
    {
        meshGas.dBuffer = dTempOutput;
        dTempOutput = 0u;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dTemp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dTempOutput)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dTempCompactedSizes)));

    _gasMeshes.emplace_back(meshGas);
}

void NerfMeshRenderer::createOptixIas()
{
    OptixAccelBuildOptions accelBuildOptions{};
    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Have one instance per mesh/NeRF.
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(_meshes.size());

    // One SBT record per GAS per ray type.
    uint32_t sbtOffset = 0;

    for (uint32_t i = 0; i < _gasMeshes.size(); ++i)
    {
        glm::mat4 meshTransform = _meshes[i]->getTransform();

        OptixInstance optixInstance{};
        optixInstance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        optixInstance.instanceId = i;
        optixInstance.visibilityMask = 255u;
        optixInstance.traversableHandle = _gasMeshes[i].handle;
        memcpy(optixInstance.transform, glm::value_ptr(glm::transpose(meshTransform)), sizeof(glm::mat3x4));
        optixInstance.sbtOffset = sbtOffset;

        sbtOffset += 1;
        optixInstances.emplace_back(optixInstance);
    }

    const size_t instancesSizeInBytes = optixInstances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_ias.dInstances), instancesSizeInBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(_ias.dInstances), optixInstances.data(), instancesSizeInBytes, cudaMemcpyHostToDevice));

    _ias.buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    _ias.buildInput.instanceArray.instances = _ias.dInstances;
    _ias.buildInput.instanceArray.numInstances = static_cast<uint32_t>(optixInstances.size());

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        _optixContext,
        &accelBuildOptions,
        &_ias.buildInput,
        1,
        &_ias.bufferSizes));

    CUdeviceptr dTempBuffer = 0u;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTempBuffer), _ias.bufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_ias.dBuffer), _ias.bufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        _optixContext,
        nullptr,
        &accelBuildOptions,
        &_ias.buildInput,
        1,
        dTempBuffer,
        _ias.bufferSizes.tempSizeInBytes,
        _ias.dBuffer,
        _ias.bufferSizes.outputSizeInBytes,
        &_ias.handle,
        nullptr,
        0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dTempBuffer)));
}

void NerfMeshRenderer::launchOptix()
{
    Params params{};
    params.image = _cudaOutputBuffer->map();
    params.imageWidth = render_width * mesh_render_size_factor;
    params.imageHeight = render_height * mesh_render_size_factor;
    params.handle = _ias.handle;
    params.camEye = {cam_pos[0], cam_pos[1], cam_pos[2]};
    auto viewMat = viewProjectionMat.data();
    params.camU = {viewMat[0], viewMat[1], viewMat[2]};
    params.camV = {viewMat[3], viewMat[4], viewMat[5]};
    params.camW = {viewMat[6], viewMat[7], viewMat[8]};
    params.lightPos = {lightPos[0], lightPos[1], lightPos[2]};
    params.bNoNerfsRendered = _nerfs.empty();
    params.dDepthBuffer = _dDepthBuffer;
    params.dFramebuffer = _dFramebuffer;

    CUdeviceptr dParam;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dParam), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(dParam), &params, sizeof(Params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(_optixPipeline, _cuStream, dParam, sizeof(Params), &_optixSbt, render_width * mesh_render_size_factor, render_height * mesh_render_size_factor, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    _cudaOutputBuffer->unmap();

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dParam)));

    // Fill image buffer with computed data.
    // _imageBuffer.data = _cudaOutputBuffer->getHostPointer();
    // _imageBuffer.width = SCREEN_WIDTH;
    // _imageBuffer.height = SCREEN_HEIGHT;
    // _imageBuffer.pixelFormat = BufferImageFormat::UNSIGNED_BYTE4;
}

void NerfMeshRenderer::cleanupOptix()
{
    cleanupOptixAccelStructs();

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_optixSbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_optixSbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_optixSbt.raygenRecord)));

    OPTIX_CHECK(optixPipelineDestroy(_optixPipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(_optixMissProgramGroup));
    OPTIX_CHECK(optixProgramGroupDestroy(_optixRaygenProgramGroup));
    OPTIX_CHECK(optixProgramGroupDestroy(_optixHitgroupMeshProgramGroup));
    OPTIX_CHECK(optixModuleDestroy(_optixModule));
    OPTIX_CHECK(optixDeviceContextDestroy(_optixContext));
}

void NerfMeshRenderer::cleanupOptixMeshGas()
{
    for (auto gasMesh : _gasMeshes)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gasMesh.dBuffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gasMesh.dVertexBuffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gasMesh.dIndexBuffer)));
    }
    _gasMeshes.clear();
}

void NerfMeshRenderer::cleanupOptixIas() const
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_ias.dBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_ias.dInstances)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_ias.dUpdateBuffer)));
}

void NerfMeshRenderer::cleanupOptixAccelStructs()
{
    cleanupOptixMeshGas();
    cleanupOptixIas();
}

void NerfMeshRenderer::cleanupMeshes()
{
    // for (auto mesh : _meshes)
    // {
    //     delete mesh;
    // }
    _meshes.clear();
}

void NerfMeshRenderer::cleanupNerfs()
{
    // for (auto nerf : _nerfs)
    // {
    //     delete nerf;
    // }
    _nerfs.clear();
}


bool NerfMeshRenderer::collide(const glm::vec3 direction, GltfNode& mesh)
{
    auto vertices = mesh.getVerticesFacingDirection(-direction);
    auto& nerf = _nerfs[0];
    auto transform = mesh.getTransform();
    tcnn::GPUMemory<float> collision_distances(vertices.size());
    std::vector<float> distances(vertices.size());
    std::vector<ngp::NerfPayload> payloads(vertices.size());
    std::vector<glm::vec3> localIntersectionPoints;
    std::vector<glm::vec3> globalIntersectionPoints;
    std::vector<glm::vec2> globalIntersectionPointsXZ;
    auto globalCentroid = glm::vec3(transform * glm::vec4(mesh.centroid(), 1.f));
    auto globalCentroidXZ = glm::vec2(globalCentroid.x, globalCentroid.z);
    memset(payloads.data(), 0, sizeof(ngp::NerfPayload) * vertices.size());

    auto copyVerticesToNerfPayloads = [&vertices, &payloads, &nerf, &direction](glm::mat4 transform){
        int i = 0;
        for (auto it = vertices.begin(); it != vertices.end(); it++, i++) {
            auto transformedVertex = glm::vec3(transform * glm::vec4(*it, 1.0f));
            payloads.at(i).alive = true;
            payloads.at(i).dir = { direction.x, direction.y, direction.z };
            payloads.at(i).idx = i;
            payloads.at(i).origin = { transformedVertex.x + 0.5f, transformedVertex.y + 0.5f, transformedVertex.z + 0.5f };
        }
        cudaMemcpy(nerf->m_nerf.tracer.rays_init().payload, payloads.data(), sizeof(ngp::NerfPayload) * vertices.size(), cudaMemcpyHostToDevice);
    };

    // Returns true if a is lexicographically before b.
    auto isLeftOf = [](const glm::vec2& a, const glm::vec2& b) {
        return (a.x < b.x || (a.x == b.x && a.y < b.y));
    };
    // Used to sort points in ccw order about a pivot.
    struct ccwSorter {
        const glm::vec2& pivot;
        ccwSorter(const glm::vec2& inPivot) : pivot(inPivot) { }
        // Positive means c is ccw from (a, b), negative cw. Zero means its collinear.
        static float ccw(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        }
        bool operator()(const glm::vec2& a, const glm::vec2& b) {
            return ccw(pivot, a, b) < 0;
        }
    };
    struct centroidDistanceSorter {
        const glm::vec2& centroid;
        centroidDistanceSorter(const glm::vec2& centroidXZ) : centroid(centroidXZ) { }
        bool operator()(const glm::vec2& a, const glm::vec2& b) {
            return glm::length(centroid - a) < glm::length(centroid - b);
        }
    };
    auto giftWrapping = [isLeftOf](std::vector<glm::vec2> v) {
        // Move the leftmost point to the beginning of our vector.
        // It will be the first point in our convext hull.
        std::swap(v[0], *std::min_element(v.begin(), v.end(), isLeftOf));
        std::vector<glm::vec2> hull;
        // Repeatedly find the first ccw point from our last hull point
        // and put it at the front of our array. 
        // Stop when we see our first point again.
        do {
            hull.push_back(v[0]);
            std::swap(v[0], *std::min_element(v.begin() + 1, v.end(), ccwSorter(v[0])));
        } while (v[0].x != hull[0].x && v[0].y != hull[0].y);

        return hull;
    };
    // The Graham scan algorithm for convex hull.
    // https://en.wikipedia.org/wiki/Graham_scan
    auto GrahamScan = [isLeftOf](std::vector<glm::vec2> v) {
        // Put our leftmost point at index 0
        std::swap(v[0], *min_element(v.begin(), v.end(), isLeftOf));
        // Sort the rest of the points in counter-clockwise order
        // from our leftmost point.
        std::sort(v.begin() + 1, v.end(), ccwSorter(v[0]));
        // Add our first three points to the hull.
        std::vector<glm::vec2> hull;
        auto it = v.begin();
        hull.push_back(*it++);
        hull.push_back(*it++);
        hull.push_back(*it++);
        while (it != v.end()) {
            // Pop off any points that make a convex angle with *it
            while (ccwSorter::ccw(*(hull.rbegin() + 1), *(hull.rbegin()), *it) >= 0) {
                hull.pop_back();
            }
            hull.push_back(*it++);
        }
        return hull;
    };
    auto pointInsideHull = [](const std::vector<glm::vec2>& hull, const glm::vec2 point) {
        auto crossProduct = [](const glm::vec2& a, const glm::vec2& b) {
            return a.x * b.y - a.y * b.x;
        };
        // Check if the point is on the same side of each edge of the convex hull
        const int numPoints = hull.size();
        for (int i = 0; i < numPoints; ++i) {
            const glm::vec2& p1 = hull[i];
            const glm::vec2& p2 = hull[(i + 1) % numPoints];
            glm::vec2 edge = p2 - p1;
            glm::vec2 pointToP1 = p1 - point;
            if (crossProduct(edge, pointToP1) < 0) {
                return false;
            }
        }
        return true;
    };




    // 0. find all mesh vertices that intersect with nerf
    collision_distances.memset(0);
    copyVerticesToNerfPayloads(transform);
    nerf->m_nerf.tracer.intersects(
        vertices.size(),
        *nerf->m_nerf_network,
        nerf->m_aabb,
        nerf->get_inference_extra_dims(nerf->m_stream.get()),
        nerf->m_nerf.density_activation,
        nerf->m_nerf.density_grid_bitfield.data(),
        collision_distances.data(),
        nerf->m_stream.get()
    );
    collision_distances.copy_to_host(distances);
    for (int i = 0; i < distances.size(); i++) {
        if (distances.at(i) > 0.f) {
            auto vertex =  *std::next(vertices.begin(), i);
            auto globalVertex = glm::vec3(transform * glm::vec4(vertex, 1.f));
            localIntersectionPoints.push_back(*std::next(vertices.begin(), i));
            globalIntersectionPoints.push_back(globalVertex);
            globalIntersectionPointsXZ.push_back({ globalVertex.x, globalVertex.z });
        }
    }

    // if 0 intersection points -> no collision: find first collision point with nerf along direction vector
    if (localIntersectionPoints.size() == 0) {
        nerf->m_nerf.tracer.collide(
            vertices.size(), 
            *nerf->m_nerf_network, 
            nerf->m_render_aabb, 
            nerf->m_render_aabb_to_local,
            nerf->m_aabb,
            nerf->m_nerf.cone_angle_constant,
            nerf->m_nerf.density_grid_bitfield.data(),
            nerf->get_inference_extra_dims(nerf->m_stream.get()),
            nerf->m_nerf.density_activation,
            collision_distances.data(),
            nerf->m_stream.get());
        collision_distances.copy_to_host(distances);
        float shortest = 10000.f;
        for (int i = 0; i < vertices.size(); i++) {
            float d = distances.at(i);
            if (d < shortest) {
                shortest = d;
            }
        }
        // move mesh by shortest collision distance
        mesh.translation += direction * shortest;
        printf("0 tip points\n");
        return false;
    }

    printf("total intersection points: %d\n", localIntersectionPoints.size());

    // check if we have a full collision
    if (localIntersectionPoints.size() >= 3) {
        // build convex hull of intersection points
        auto hull = GrahamScan(globalIntersectionPointsXZ);
        printf("centroid %f %f\n", globalCentroid.x, globalCentroid.z);
        printf("hull ");
        for (auto p : hull)
            printf("[%f %f] ", p.x, p.y);
        printf("\n");
        // check if centroid is inside hull
        if (pointInsideHull(hull, { globalCentroid.x, globalCentroid.z }))
            return true;
    }

    // we don't have a full collision --> either rotation around 1 point or around 2 point axis 


    // first tip is the vertex with the smallest xz distance to the centroid
    // second tip is chosen based on following criteria:
    // 1. must have a minimum distance to first tip
    // 2. connecting vector must not be too parallel to vector connecting middle point with centroid
    // 3. connecting vector must be the most orthogonal with respect to vector connecting middle point with centroid 

    auto itClosestPoint = std::min_element(globalIntersectionPointsXZ.begin(), globalIntersectionPointsXZ.end(), centroidDistanceSorter({ globalCentroid.x, globalCentroid.z }));
    int idxClosestPoint = std::distance(globalIntersectionPointsXZ.begin(), itClosestPoint);
    auto firstTipXZ = *itClosestPoint;
    glm::vec3 t1 = localIntersectionPoints.at(idxClosestPoint), t2;
    auto centroid = mesh.centroid();
    bool foundSecond = false;
    float bestSecondPointAngle = 42.f;

    printf("first tip %f %f\n", firstTipXZ.x, firstTipXZ.y);

    for (auto it = globalIntersectionPointsXZ.begin(); it != globalIntersectionPointsXZ.end(); it++) {
        auto v = *it - firstTipXZ;
        // 1. must have a minimum distance to first tip
        if (glm::length(v) < 0.1)
            continue;
        auto middle = (firstTipXZ + *it) / 2.f;
        auto toCentroid = globalCentroidXZ - middle;
        float angle = acosf(glm::dot(v, toCentroid) / (glm::length(v) * glm::length(toCentroid)));
        float diffToRightAngle = fabs(angle - M_PI_2);
        float proj = glm::dot(globalCentroidXZ - firstTipXZ, v) / glm::length2(v);
        bool centroidBetweenPoints = proj > 0 && proj < 1;
        // 2. connecting vector must not be too parallel to vector connecting middle point with centroid
        if (!centroidBetweenPoints && diffToRightAngle > M_PI_2 / 2)
            continue;
        foundSecond = true;
        // 3. connecting vector must be the most orthogonal with respect to vector connecting middle point with centroid 
        if (diffToRightAngle < bestSecondPointAngle) {
            bestSecondPointAngle = diffToRightAngle;
            t2 = localIntersectionPoints.at(std::distance(globalIntersectionPointsXZ.begin(), it));
        }
    }
    if (foundSecond)
        printf("second tip %f %f %f\n", t2.x, t2.y, t2.z);

    copyVerticesToNerfPayloads(mesh.getTransform());

    // one tip point
    if (!foundSecond) {
        auto rotPoint = t1;
        auto rotAxis = glm::normalize(glm::cross(glm::normalize(mesh.centroid() - rotPoint), direction));
        mesh.RotateAroundAxis(rotAxis, rotPoint, 0.5f);
        printf("one tip point\n");
        return false;
    }

    // two tip points
    auto rotPoint = t1;
    auto rotAxis = glm::normalize(t2 - t1);
    int rotationDirection = glm::cross(glm::normalize(mesh.centroid() - rotPoint), rotAxis).y > 0 ? 1 : -1;
    mesh.RotateAroundAxis(rotAxis, rotPoint, rotationDirection * 0.5f);
    printf("two tip points\n");
    return false;
}
