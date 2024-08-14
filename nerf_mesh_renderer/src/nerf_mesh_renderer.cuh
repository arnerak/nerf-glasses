#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <optix.h>

#include "optix/optix_scene.cuh"
#include "cuda_output_buffer.cuh"
#include "camera.h"
#include "gltf_scene.h"
#include "ngp/testbed.cuh"


template<typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct ImGuiOptions {
    double fps = 0.0;
    bool bShowDemoWindow = false;
};

struct CursorControls {
    bool bIsRmbPressed = false;
    double cursorPosX = 0.0;
    double cursorPosY = 0.0;
};

struct MeshGas {
    OptixTraversableHandle handle = 0u;
    CUdeviceptr dBuffer = 0u;
    CUdeviceptr dVertexBuffer = 0u;
    CUdeviceptr dIndexBuffer = 0u;
};

struct Ias {
    OptixTraversableHandle handle = 0u;
    OptixAccelBufferSizes bufferSizes = {};
    OptixBuildInput buildInput = {};
    CUdeviceptr dBuffer = 0u;
    CUdeviceptr dInstances = 0u;
    CUdeviceptr dUpdateBuffer = 0u;
};

class NerfMeshRenderer {
public:
    uint32_t SCREEN_WIDTH = 1280u;
    uint32_t SCREEN_HEIGHT = 720u;

    NerfMeshRenderer(int width, int height);
    ~NerfMeshRenderer()
    {
        cleanup();
    }
    void cleanup();
    void run();
    bool frame();

    GltfScene* loadMesh(
        const std::string& filename, 
        const Eigen::Vector3f& t = { 0.f, 0.f, 0.f }, 
        const Eigen::Vector3f& s = { 1.f, 1.f, 1.f }, 
        const Eigen::Vector4f& r = { 0.f, 0.f, 0.f, 1.f });
    ngp::Testbed* loadNerf(const std::string& filename);

    Eigen::Matrix<float, 3, 4> viewProjectionMat;
    

    void orbit(float delta_azimuth, float delta_polar, float delta_zoom);


    void removeFloaties();


private:
    bool _bIsInitialized = false;
    GLFWwindow* _window = nullptr;
    CursorControls _cursorControls = {};
    ImGuiOptions _imGuiOptions = {};

	// camera variables
	float viewMat[16];
	float cam_pos[3]  { 0.f, 0.f, 2.f };
	float cam_look[3] { 0.f, -0.000001f, -0.999999f };
	const float up[3] { 0, 1, 0 };
    float cam_pivot[3] = { 0.f, 0.f, 0.f };
    double scroll_offset = 0.0;
    float lightPos[3] { 1, 1, 1 };
    
    Camera* _camera = nullptr;
    std::vector<GltfScene*> _meshes = {};
    std::vector<ngp::Testbed*> _nerfs = {};

    uint32_t _glProgram = 0u;
    uint32_t _glVao = 0u;
    uint32_t _glPbo = 0u;
    uint32_t _glRenderTex = 0u;

    uint32_t numberFrames = 0;
    double lastTime;
    double lastFpsTime;
    uint32_t render_width;
    uint32_t render_height;
    float render_size_factor = 1.f;
    int mesh_render_size_factor = 2.f;
    CudaOutputBuffer<uchar4>* _cudaOutputBuffer = nullptr;
    ImageBuffer _imageBuffer = {};
    std::shared_ptr<ngp::GLTexture> _ngpRenderTexture;

    Eigen::Array4f* _dFramebuffer = nullptr;
    float* _dDepthBuffer = nullptr;

    CUstream _cuStream = nullptr;

    OptixDeviceContext _optixContext = nullptr;
    OptixModule _optixModule = nullptr;
    OptixProgramGroup _optixRaygenProgramGroup = nullptr;
    OptixProgramGroup _optixMissProgramGroup = nullptr;
    OptixProgramGroup _optixHitgroupMeshProgramGroup = nullptr;
    OptixPipeline _optixPipeline = nullptr;
    OptixShaderBindingTable _optixSbt = {};
    std::vector<MeshGas> _gasMeshes = {};
    Ias _ias = {};

    void update();
    void render_frame();
    void gui();
	bool handleInput(float deltaTime);
    void updateModelViewProj();

    std::vector<uint8_t> dumpDensityGrid();
    void loadDensityGrid(const std::vector<uint8_t>& grid);
    void dumpDensityGrid(const std::string& filename);
    void loadDensityGrid(const std::string& filename);

    bool collide(const glm::vec3 direction, GltfNode& mesh);

    void initOptix();
    void createOptixContext();
    void createOptixModule(OptixPipelineCompileOptions& pipelineCompileOptions);
    void createOptixProgramGroups();
    void createOptixPipeline(OptixPipelineCompileOptions& pipelineCompileOptions);
    void createOptixSbt();
    void createOptixMeshGas(const GltfScene* mesh);
    void createOptixIas();
    void launchOptix();
    void cleanupOptix();
    void cleanupOptixMeshGas();
    void cleanupOptixIas() const;
    void cleanupOptixAccelStructs();

    void cleanupMeshes();
    void cleanupNerfs();

    void keyCallback(int key, int scancode, int action, int mods);
    void cursorPosCallback(double xPos, double yPos);
    void mouseButtonCallback(int button, int action, int mods);
    void scrollCallback(double xOffset, double yOffset);

    // Mark GLFW callbacks as friends to allow them to delegate their calls to NerfMeshRenderer's private functions.
    friend void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    friend void cursorPosCallback(GLFWwindow* window, double xPos, double yPos);
    friend void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    friend void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
};