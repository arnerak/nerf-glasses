#include "nerf_mesh_renderer.cuh"

int main() {
#if !defined(NDEBUG)
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif

    try {
        NerfMeshRenderer renderer{ 1280, 720 };


        //auto mesh = renderer.loadMesh("../assets/meshes/glasses/glasses.gltf");
        //mesh->nodes[0]->scale = { 4.f, 4.f, 4.f };
        //positionGlasses(*_meshes.at(0)->nodes[0], 0.f);
        //renderer.loadNerf("../assets/nerfs/tristan2.msgpack");

        renderer.run();
    }
    catch(const std::exception& e) {
        spdlog::error("Crash: {}", e.what());
    }

    return 0;
}
