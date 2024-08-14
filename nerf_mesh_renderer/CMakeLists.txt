cmake_minimum_required(VERSION 3.21)
project(nerf_mesh_renderer LANGUAGES CUDA CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# == CUDA Compiler =====================================================================================================

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")   # Not really working...
#list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_86,code=sm_86")

# Have to set this manually because the "--generate-code" option does not work with CLion, apparently.
# It's an issue similar to this: https://youtrack.jetbrains.com/issue/CPP-20384
set(CMAKE_CUDA_ARCHITECTURES "86")

# == C++ Compiler ======================================================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

# == Dependencies ======================================================================================================

find_package(OptiX REQUIRED)


# set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
# set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
# set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
# set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
# set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
# set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
# set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
# set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

# add_subdirectory(dependencies/GLFW)
# set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

find_package(Python 3.7 COMPONENTS Interpreter Development)
if (Python_FOUND)
	add_subdirectory("dependencies/pybind11")
endif()

add_subdirectory("dependencies")

list(APPEND NMR_INCLUDE_DIRS
        "dependencies"
        "dependencies/eigen"
        "dependencies/filesystem"
        "dependencies/GLFW/include"
        "dependencies/tiny-cuda-nn/include"
        "dependencies/glm"
        "dependencies/pybind11"
        "dependencies/tiny-cuda-nn/dependencies"    # json.hpp
        "${OptiX_INCLUDE}"
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

list(APPEND NMR_LIBRARIES
        glfw
        glad
        #glm
        imgui
        tiny-cuda-nn
        mikktspace
)

if (NOT MSVC)
    list(APPEND NMR_LIBRARIES ${CMAKE_DL_LIBS})
endif()

message(STATUS "NMR_LIBRARIES: ${NMR_LIBRARIES}")

message(STATUS "Optix ${OptiX_INCLUDE}")

add_library(optix_program OBJECT
        src/optix/optix_util.cuh
        src/optix/optix_scene.cuh
        src/optix/optix_scene.cu
)

target_compile_definitions(optix_program PUBLIC -DTCNN_MIN_GPU_ARCH=0)
get_target_property(TCNN_INCLUDE_DIRECTORIES tiny-cuda-nn INCLUDE_DIRECTORIES)

set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")
target_include_directories(optix_program PUBLIC ${NMR_INCLUDE_DIRS} ${TCNN_INCLUDE_DIRECTORIES})

# OptiX programs will be compiled as PTX and packaged
# as headers to be included from the binary dir.
list(APPEND NMR_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
if (NOT bin_to_c)
    message(FATAL_ERROR
            "bin2c not found:\n"
            "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
            "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
    )
endif()

add_custom_command(
        OUTPUT "${OPTIX_PTX_HEADER}"
        COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${bin_to_c}"
        "-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
        "-DOUTPUT=${OPTIX_PTX_HEADER}"
        -P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
        VERBATIM
        DEPENDS optix_program
        COMMENT "Converting PTX files to a C header"
)

# == Executable ========================================================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

set(NMR_SOURCE_FILES
        src/common.h
        src/common.cpp
        src/camera.h
        src/camera.cpp
        src/gltf_scene.h
        src/gltf_mikktspace_handler.cpp
        src/gltf_scene.cpp
        src/cuda_texture.cuh
        src/cuda_texture.cu
        src/cuda_output_buffer.cuh
        src/nerf_mesh_renderer.cuh
        src/nerf_mesh_renderer.cu
        src/ngp/adam_optimizer.h
        src/ngp/json_binding.h
        src/ngp/nerf.cuh
        src/ngp/nerf_loader.cuh
        src/ngp/nerf_loader.cu
        src/ngp/nerf_network.cuh
        src/ngp/bounding_box.cuh
        src/ngp/random_val.cuh
        src/ngp/render_buffer.cuh
        src/ngp/render_buffer.cu
        src/ngp/trainable_buffer.cuh
        src/ngp/thread_pool.h
        src/ngp/thread_pool.cpp
        src/ngp/ngp_common.cuh
        src/ngp/testbed.cuh
        src/ngp/testbed.cu
        "${OPTIX_PTX_HEADER}"
)

add_library(nmr STATIC ${NMR_SOURCE_FILES})
set_target_properties(nmr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(nmr PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(nmr PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(nmr PUBLIC ${NMR_INCLUDE_DIRS})
target_link_libraries(nmr PUBLIC ${NMR_LIBRARIES})

add_executable(nerf_mesh_renderer src/main.cu)
target_link_libraries(nerf_mesh_renderer PRIVATE nmr)

if (Python_FOUND)
	add_library(pynmr SHARED src/python_api.cu)
	set_target_properties(pynmr PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
	target_link_libraries(pynmr PRIVATE nmr PUBLIC ${PYTHON_LIBRARIES} pybind11::module)
	target_compile_definitions(pynmr PUBLIC -DNGP_PYTHON)
	pybind11_extension(pynmr)
endif()
