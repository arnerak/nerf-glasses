#include "gltf_scene.h"

#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tinygltf/tiny_gltf.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "common.h"

GltfScene* GltfLoader::load(const std::string& filename) {
    tinygltf::Model model = loadTinygltfModel(filename);

    auto gltfScene = new GltfScene();

    gltfScene->name = model.scenes[model.defaultScene].name;

    std::vector<GltfNode*> gltfNodes;
    for(const auto nodeIndex : model.scenes[model.defaultScene].nodes) {
        auto gltfNode = traverse(model, nodeIndex);
        gltfNodes.emplace_back(gltfNode);
    }

    gltfScene->nodes = gltfNodes;

    return gltfScene;
}

tinygltf::Model GltfLoader::loadTinygltfModel(const std::string& filename) {
    bool bIsBinary = filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".glb") == 0;

    tinygltf::Model model;
    std::string errorLog;
    std::string warnLog;
    bool bLoadResult;

    tinygltf::TinyGLTF loader;
    if(bIsBinary) {
        bLoadResult = loader.LoadBinaryFromFile(&model, &errorLog, &warnLog, filename);
    }
    else {
        bLoadResult = loader.LoadASCIIFromFile(&model, &errorLog, &warnLog, filename);
    }

    if(!errorLog.empty()) {
        spdlog::error("glTF load error: {}", errorLog);
    }

    if(!warnLog.empty()) {
        spdlog::warn("glTF load warning: {}", warnLog);
    }

    if(!bLoadResult) {
        spdlog::error("Loading of glTF file {} failed", filename);
        throw std::runtime_error("Aborted glTF loading due to failure.");
    }

    return model;
}

GltfNode* GltfLoader::traverse(const tinygltf::Model& model, int nodeIndex) {
    auto gltfNode = new GltfNode();

    tinygltf::Node node = model.nodes[nodeIndex];

    gltfNode->name = node.name;

    int meshIndex = node.mesh;
    if(meshIndex != -1) {
        gltfNode->mesh = loadMesh(model, meshIndex);
    }

    if(!node.matrix.empty()) {
        const glm::mat4 matrix = glm::make_mat4(node.matrix.data());

        // Temporary handles that are needed for glm::decompose().
        glm::vec3 skew;
        glm::vec4 perspective;

        // Apparently glm::decompose returns a wrong quaternion. See https://stackoverflow.com/a/40024726/4303296.
        glm::decompose(matrix, gltfNode->scale, gltfNode->rotation, gltfNode->translation, skew, perspective);
        gltfNode->rotation = glm::conjugate(gltfNode->rotation);
    }
    else {
        glm::vec3 t(0.0f);
        if(!node.translation.empty()) {
            t = glm::make_vec3(node.translation.data());
        }
        gltfNode->translation = t;

        glm::quat r(1.0f, 0.0f, 0.0f, 0.0f);
        if(!node.rotation.empty()) {
            r = glm::quat(
                    static_cast<float>(node.rotation[3]),
                    static_cast<float>(node.rotation[0]),
                    static_cast<float>(node.rotation[1]),
                    static_cast<float>(node.rotation[2])
            );
        }
        gltfNode->rotation = r;

        glm::vec3 s(1.0f);
        if(!node.scale.empty()) {
            s = glm::make_vec3(node.scale.data());
        }
        gltfNode->scale = s;
    }

    for(const auto childIndex : node.children) {
        GltfNode* child = traverse(model, childIndex);
        gltfNode->children.emplace_back(child);
    }

    return gltfNode;
}

GltfMesh *GltfLoader::loadMesh(const tinygltf::Model &model, int meshIndex) {
    auto gltfMesh = new GltfMesh();

    for(const auto& meshPrimitive : model.meshes[meshIndex].primitives) {
        auto gltfMeshPrimitive = new GltfMeshPrimitive();

        if(meshPrimitive.indices != -1) {
            processAccessor(model, meshPrimitive.indices, AccessorPrimitive::INDICES, gltfMeshPrimitive);
        }

        const auto& attributes = meshPrimitive.attributes;
        if(attributes.count("POSITION") != 0) {
            processAccessor(model, attributes.at("POSITION"), AccessorPrimitive::POSITION, gltfMeshPrimitive);
        }
        if(attributes.count("NORMAL") != 0) {
            processAccessor(model, attributes.at("NORMAL"), AccessorPrimitive::NORMAL, gltfMeshPrimitive);
        }
        if(attributes.count("TANGENT") != 0) {
            processAccessor(model, attributes.at("TANGENT"), AccessorPrimitive::TANGENT, gltfMeshPrimitive);
        }
        if(attributes.count("TEXCOORD_0") != 0) {
            processAccessor(model, attributes.at("TEXCOORD_0"), AccessorPrimitive::TEXCOORD_0, gltfMeshPrimitive);
        }

        if(meshPrimitive.material != -1) {
            gltfMeshPrimitive->material = loadMaterial(model, meshPrimitive.material);
        }

        gltfMesh->meshPrimitives.emplace_back(gltfMeshPrimitive);
    }

    auto mikkTSpaceHandler = MikkTSpaceHandler();
    for(const auto& meshPrimitive : gltfMesh->meshPrimitives) {
        if(meshPrimitive->tangents.empty()) {
            meshPrimitive->tangents.resize(meshPrimitive->normals.size());
            mikkTSpaceHandler.calcTangents(meshPrimitive);
        }
    }

    return gltfMesh;
}

GltfMaterial *GltfLoader::loadMaterial(const tinygltf::Model &model, int materialIndex) {
    auto gltfMaterial = new GltfMaterial();

    auto readImage = [&model](int idx, bool bSrgb) -> CudaTexture* {
        const auto& image = model.images[idx];
        return new CudaTexture(image.width, image.height, image.image, bSrgb);
    };

    const auto& material = model.materials[materialIndex];

    gltfMaterial->name = material.name;

    gltfMaterial->emissiveFactor = glm::vec3(
            static_cast<float>(material.emissiveFactor[0]),
            static_cast<float>(material.emissiveFactor[1]),
            static_cast<float>(material.emissiveFactor[2])
    );

    if(material.emissiveTexture.index != -1) {
        gltfMaterial->emissiveTexture = readImage(material.emissiveTexture.index, true);
    }

    {
        const auto& pbr = material.pbrMetallicRoughness;

        gltfMaterial->baseColor = glm::vec4(
                static_cast<float>(pbr.baseColorFactor[0]),
                static_cast<float>(pbr.baseColorFactor[1]),
                static_cast<float>(pbr.baseColorFactor[2]),
                static_cast<float>(pbr.baseColorFactor[3])
        );

        if(pbr.baseColorTexture.index != -1) {
            gltfMaterial->baseColorTexture = readImage(pbr.baseColorTexture.index, true);
        }

        gltfMaterial->metallicFactor = static_cast<float>(pbr.metallicFactor);
        gltfMaterial->roughnessFactor = static_cast<float>(pbr.roughnessFactor);

        if(pbr.metallicRoughnessTexture.index != -1) {
            gltfMaterial->metallicRoughnessTexture = readImage(pbr.metallicRoughnessTexture.index, false);
        }
    }

    gltfMaterial->normalScale = static_cast<float>(material.normalTexture.scale);
    if(material.normalTexture.index != -1) {
        gltfMaterial->normalTexture = readImage(material.normalTexture.index, false);
    }

    gltfMaterial->occlusionStrength = static_cast<float>(material.occlusionTexture.strength);
    if(material.occlusionTexture.index != -1) {
        gltfMaterial->occlusionTexture = readImage(material.occlusionTexture.index, false);
    }

    return gltfMaterial;
}

void GltfLoader::processAccessor(
        const tinygltf::Model &model,
        int accessorIndex,
        AccessorPrimitive accessorPrimitive,
        GltfMeshPrimitive* gltfMeshPrimitive
) {
    const tinygltf::Accessor accessor = model.accessors[accessorIndex];

    if(accessor.bufferView != -1) {
        const tinygltf::BufferView bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer buffer = model.buffers[bufferView.buffer];

        const std::vector<uint8_t> bytes = readBytes(model, accessor);

        // Write into appropriate field (indices, positions, etc.).
        switch(accessorPrimitive) {
            case AccessorPrimitive::POSITION:
            {
                for(int i = 0; i < bytes.size(); i += sizeof(glm::vec3)) {
                    glm::vec3 position;
                    memcpy(glm::value_ptr(position), &bytes.data()[i], sizeof(glm::vec3));
                    gltfMeshPrimitive->positions.emplace_back(position);
                }
                break;
            }
            case AccessorPrimitive::NORMAL:
            {
                for(int i = 0; i < bytes.size(); i += sizeof(glm::vec3)) {
                    glm::vec3 normal;
                    memcpy(glm::value_ptr(normal), &bytes.data()[i], sizeof(glm::vec3));
                    gltfMeshPrimitive->normals.emplace_back(normal);
                }
                break;
            }
            case AccessorPrimitive::TANGENT:
            {
                for(int i = 0; i < bytes.size(); i += sizeof(glm::vec4)) {
                    glm::vec4 tangent;
                    memcpy(glm::value_ptr(tangent), &bytes.data()[i], sizeof(glm::vec4));
                    gltfMeshPrimitive->tangents.emplace_back(tangent);
                }
                break;
            }
            case AccessorPrimitive::TEXCOORD_0:
                for(int i = 0; i < bytes.size(); i += sizeof(glm::vec2)) {
                    glm::vec2 texCoord;
                    memcpy(glm::value_ptr(texCoord), &bytes.data()[i], sizeof(glm::vec2));
                    gltfMeshPrimitive->texCoords.emplace_back(texCoord);
                }
                break;
            case AccessorPrimitive::INDICES:
                SWITCH_FALLTHROUGH;
            default:
            {
                for(int i = 0; i < bytes.size(); i += sizeof(uint16_t)) {
                    // For now assume indices are always of type ushort.
                    uint8_t indicesBytes[2] = { bytes[i], bytes[i + 1] };
                    uint16_t index;
                    memcpy(&index, indicesBytes, sizeof(uint16_t));
                    gltfMeshPrimitive->indices.emplace_back(index);
                }
            }
        }
    }
    else {
        spdlog::debug("TODO: bufferView of accessor {} is null", accessorIndex);
    }
}

std::vector<uint8_t> GltfLoader::readBytes(const tinygltf::Model &model, const tinygltf::Accessor& accessor) {
    const tinygltf::BufferView bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer buffer = model.buffers[bufferView.buffer];

    const size_t elementSize = computeElementSize(accessor);
    const size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
    const size_t byteStride = bufferView.byteStride == 0 ? elementSize : bufferView.byteStride;
    const size_t endIndex = byteOffset + accessor.count * byteStride;

    // Retrieve bytes in sequential order (accounting for interleaved vs. per-attribute format).
    std::vector<uint8_t> bytes;
    for(size_t i = byteOffset; i < endIndex; i += byteStride) {
        for(size_t j = 0; j < elementSize; ++j) {
            bytes.emplace_back(buffer.data[i + j]);
        }
    }

    return bytes;
}

size_t GltfLoader::computeElementSize(const tinygltf::Accessor& accessor) {
    size_t numberOfComponents;
    switch(accessor.type) {
        case TINYGLTF_TYPE_SCALAR:
            numberOfComponents = 1;
            break;
        case TINYGLTF_TYPE_VEC2:
            numberOfComponents = 2;
            break;
        case TINYGLTF_TYPE_VEC3:
            numberOfComponents = 3;
            break;
        case TINYGLTF_TYPE_VEC4:
            numberOfComponents = 4;
            break;
        case TINYGLTF_TYPE_MAT2:
            numberOfComponents = 2 * 2;
            break;
        case TINYGLTF_TYPE_MAT3:
            numberOfComponents = 3 * 3;
            break;
        case TINYGLTF_TYPE_MAT4:
            SWITCH_FALLTHROUGH;
        default:
            numberOfComponents = 4 * 4;
            break;
    }

    size_t componentSizeInBytes;
    switch(accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            componentSizeInBytes = sizeof(int8_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            componentSizeInBytes = sizeof(uint8_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            componentSizeInBytes = sizeof(int16_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            componentSizeInBytes = sizeof(uint16_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_INT:
            componentSizeInBytes = sizeof(int32_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            componentSizeInBytes = sizeof(uint32_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            SWITCH_FALLTHROUGH;
        default:
            componentSizeInBytes = sizeof(float);
            break;
    }

    return numberOfComponents * componentSizeInBytes;
}


void GltfNode::RotateAroundAxis(const glm::vec3& axis, const glm::vec3& localPoint, float angleDegrees) {
    // Rotate the mesh around the specified axis in local space
    glm::quat rotationQuaternion = glm::angleAxis(glm::radians(angleDegrees), axis);
    glm::vec3 wtf = glm::rotate(rotation, scale * localPoint);
    translation += wtf - glm::rotate(rotationQuaternion, wtf);
    rotation = rotationQuaternion * rotation;
}

