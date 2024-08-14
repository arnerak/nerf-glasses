#pragma once

#include <tinygltf/tiny_gltf.h>
#include <MikkTSpace/mikktspace.h>
#include <spdlog/spdlog.h>
#include <glm/mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>

#include <optional>
#include <string>
#include <utility>
#include <stack>
#include <vector>
#include <unordered_set>
#include <iostream>

#include "cuda_texture.cuh"

/**
 * Note:    this loader only supports a very small subset of the glTF core specification.
 *          There is no support for .glb files, animations, and blending, among other things.
 *          Furthermore, the renderer (see optix_scene.cu) also makes some simplifications, e.g. it assumes that
 *          normals and texture coordinates exist.
 *
 *          Working models include DamagedHelmet, Suzanne, BarramundiFish, WaterBottle, and Box With Spaces,
 *          which come shipped with this project.
 *          Other models are likely to not work, and if support for those is desired, the loader will have to be
 *          expanded in functionality. (See OptiX samples for a proper loader.)
 */

enum class AccessorPrimitive {
    INDICES,
    POSITION,
    NORMAL,
    TANGENT,
    TEXCOORD_0
};

struct GltfMaterial {
    std::string name;

    glm::vec3 emissiveFactor{ 0.0f };
    CudaTexture* emissiveTexture = nullptr;

    glm::vec4 baseColor{ 1.0f };
    CudaTexture* baseColorTexture = nullptr;

    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    CudaTexture* metallicRoughnessTexture = nullptr;

    float normalScale = 1.0f;
    CudaTexture* normalTexture = nullptr;

    float occlusionStrength = 1.0f;
    CudaTexture* occlusionTexture = nullptr;

    ~GltfMaterial() {
        delete occlusionTexture;
        delete normalTexture;
        delete metallicRoughnessTexture;
        delete baseColorTexture;
        delete emissiveTexture;
    }
};

struct GltfMeshPrimitive {
    GltfMaterial* material;
    std::vector<uint16_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> texCoords;

    ~GltfMeshPrimitive() {
        delete material;
    }
};

struct GltfMesh {
    std::vector<GltfMeshPrimitive*> meshPrimitives;

    ~GltfMesh() {
        for(auto meshPrimitive : meshPrimitives) {
            delete meshPrimitive;
        }
    }
};

struct GltfNode {

    struct KeyFuncs
    {
        size_t operator()(const glm::vec3& k)const
        {
            return std::hash<float>()(k.x) ^ std::hash<float>()(k.y) ^ std::hash<float>()(k.z);
        }

        bool operator()(const glm::vec3& a, const glm::vec3& b)const
        {
            return glm::all(glm::epsilonEqual(a, b, 0.01f));
        }
    };

    std::string name;
    GltfMesh* mesh;
    std::vector<GltfNode*> children;
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    std::unordered_set<glm::vec3, KeyFuncs> verticesFacingDirection;
    glm::vec3 lastDirectionVec;


    ~GltfNode() {
        delete mesh;
        for(auto child : children) {
            delete child;
        }
    }

    [[nodiscard]] glm::mat4 getTransform() const {
        const glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
        const glm::mat4 r = glm::toMat4(rotation);
        const glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void RotateAroundAxis(const glm::vec3& axis, const glm::vec3& worldPoint, float angleDegrees);

    glm::vec3 centroid() const {
        float meshVolume = 0.f;
        glm::vec3 temp { 0.f, 0.f, 0.f };
        for (int i = 0; i < mesh->meshPrimitives[0]->positions.size(); i += 3) {
            auto& v1 = mesh->meshPrimitives[0]->positions.at(i + 0);
            auto& v2 = mesh->meshPrimitives[0]->positions.at(i + 1);
            auto& v3 = mesh->meshPrimitives[0]->positions.at(i + 2);
            auto center = (v1 + v2 + v3) / 4.f;
            auto volume = glm::dot(v1, glm::cross(v2, v3)) / 6.f;
            meshVolume += volume;
            temp += center * volume;
        }
        auto centroid = temp / meshVolume;
        return centroid;
    }

    auto getVerticesFacingDirection(const glm::vec3& dir) {
        if (glm::all(glm::epsilonEqual(dir, lastDirectionVec, 0.001f))) {
            return verticesFacingDirection;
        }
        lastDirectionVec = dir;
        verticesFacingDirection.clear();

        auto r = glm::toMat3(rotation);

        for (auto primitive : mesh->meshPrimitives) {
            for (int i = 0; i < primitive->normals.size(); i++) {
                auto normal = r * primitive->normals.at(i);
                if (glm::dot(normal, dir) < 0) {
                    verticesFacingDirection.insert(primitive->positions.at(i));
                }
            }
        }

        for (auto child : children) {
            auto vertices = child->getVerticesFacingDirection(dir);
            verticesFacingDirection.insert(vertices.begin(), vertices.end());
        }

        return verticesFacingDirection;
    }
};

struct GltfScene {
    std::string name;
    std::vector<GltfNode*> nodes;

    ~GltfScene() {
        for(auto node : nodes) {
            delete node;
        }
    }

    [[nodiscard]] std::string getName() const {
        if(!name.empty()) {
            return name;
        }
        else if(!nodes.empty() && !nodes[0]->name.empty()){
            return nodes[0]->name;
        }
        else {
            return "Scene";
        }
    }

    [[nodiscard]] std::vector<GltfMeshPrimitive*> getMeshPrimitives() const {
        std::vector<GltfMeshPrimitive*> meshPrimitives;

        std::stack<GltfNode*> nodesStack;
        for(const auto node : nodes) {
            nodesStack.push(node);
        }

        while(!nodesStack.empty()) {
            GltfNode* node = nodesStack.top();
            GltfMesh* mesh = node->mesh;
            std::vector<GltfNode*> children = node->children;
            nodesStack.pop();

            // Retrieve all meshes from the current node.
            if(mesh != nullptr && !mesh->meshPrimitives.empty()) {
                for(const auto meshPrimitive : mesh->meshPrimitives) {
                    meshPrimitives.emplace_back(meshPrimitive);
                }
            }

            // Stack up child nodes to retrieve their meshes.
            for(const auto child : children) {
                nodesStack.push(child);
            }
        }

        return meshPrimitives;
    }

    [[nodiscard]] glm::mat4 getTransform() const {
        if(!nodes.empty()) {
            return nodes[0]->getTransform();
        }

        return glm::mat4(1.0f);
    }
};

class GltfLoader {
public:
    static GltfScene* load(const std::string& filename);

private:
    static tinygltf::Model loadTinygltfModel(const std::string& filename);
    static GltfNode* traverse(const tinygltf::Model& model, int nodeIndex);
    static GltfMesh* loadMesh(const tinygltf::Model& model, int meshIndex);
    static GltfMaterial* loadMaterial(const tinygltf::Model& model, int materialIndex);
    static void processAccessor(const tinygltf::Model& model, int accessorIndex, AccessorPrimitive accessorPrimitive, GltfMeshPrimitive* gltfMeshPrimitive);
    static std::vector<uint8_t> readBytes(const tinygltf::Model& model, const tinygltf::Accessor& accessor);
    static size_t computeElementSize(const tinygltf::Accessor& accessor);
};

// Approach from https://www.turais.de/using-mikktspace-in-your-project/.
class MikkTSpaceHandler {
public:
    MikkTSpaceHandler();
    void calcTangents(GltfMeshPrimitive* gltfMeshPrimitive);

private:
    SMikkTSpaceInterface _interface = {};
    SMikkTSpaceContext _context = {};

    static int getVertexIndex(const SMikkTSpaceContext* context, int faceIndex, int vertexIndex);
    static int getNumFaces(const SMikkTSpaceContext* context);
    static int getNumVerticesOfFace(const SMikkTSpaceContext* context, int faceIndex);
    static void getPosition(const SMikkTSpaceContext* context, float outPos[], int faceIndex, int vertexIndex);
    static void getNormal(const SMikkTSpaceContext* context, float outNormal[], int faceIndex, int vertexIndex);
    static void getTexCoord(const SMikkTSpaceContext* context, float outUv[], int faceIndex, int vertexIndex);
    static void setTSpaceBasic(const SMikkTSpaceContext* context, const float tangentU[], float fSign, int faceIndex, int vertexIndex);
};