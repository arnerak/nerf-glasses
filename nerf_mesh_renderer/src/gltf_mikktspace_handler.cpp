#include "gltf_scene.h"

MikkTSpaceHandler::MikkTSpaceHandler() {
    _interface.m_getNumFaces = getNumFaces;
    _interface.m_getNumVerticesOfFace = getNumVerticesOfFace;
    _interface.m_getPosition = getPosition;
    _interface.m_getNormal = getNormal;
    _interface.m_getTexCoord = getTexCoord;
    _interface.m_setTSpaceBasic = setTSpaceBasic;

    _context.m_pInterface = &_interface;
}

void MikkTSpaceHandler::calcTangents(GltfMeshPrimitive* gltfMeshPrimitive) {
    _context.m_pUserData = gltfMeshPrimitive;
    genTangSpaceDefault(&_context);
}

int MikkTSpaceHandler::getVertexIndex(const SMikkTSpaceContext* context, int faceIndex, int vertexIndex) {
    const auto meshPrimitive = reinterpret_cast<GltfMeshPrimitive*>(context->m_pUserData);
    int faceSize = getNumVerticesOfFace(context, faceIndex);
    int indicesIndex = (faceIndex * faceSize) + vertexIndex;
    return meshPrimitive->indices[indicesIndex];
}

int MikkTSpaceHandler::getNumFaces(const SMikkTSpaceContext *context) {
    const auto meshPrimitive = reinterpret_cast<GltfMeshPrimitive*>(context->m_pUserData);
    return static_cast<int>(meshPrimitive->indices.size() / 3);
}

int MikkTSpaceHandler::getNumVerticesOfFace(const SMikkTSpaceContext* context, int faceIndex) {
    return 3;
}

void MikkTSpaceHandler::getPosition(const SMikkTSpaceContext* context, float* outPos, int faceIndex, int vertexIndex) {
    const auto meshPrimitive = reinterpret_cast<GltfMeshPrimitive*>(context->m_pUserData);
    int index = getVertexIndex(context, faceIndex, vertexIndex);
    glm::vec3 position = meshPrimitive->positions[index];
    outPos[0] = position.x;
    outPos[1] = position.y;
    outPos[2] = position.z;
}

void MikkTSpaceHandler::getNormal(const SMikkTSpaceContext* context, float* outNormal, int faceIndex, int vertexIndex) {
    const auto meshPrimitive = reinterpret_cast<GltfMeshPrimitive*>(context->m_pUserData);
    int index = getVertexIndex(context, faceIndex, vertexIndex);
    glm::vec3 normal = meshPrimitive->normals[index];
    outNormal[0] = normal.x;
    outNormal[1] = normal.y;
    outNormal[2] = normal.z;
}

void MikkTSpaceHandler::getTexCoord(const SMikkTSpaceContext* context, float* outUv, int faceIndex, int vertexIndex) {
    const auto meshPrimitive = reinterpret_cast<GltfMeshPrimitive*>(context->m_pUserData);
    int index = getVertexIndex(context, faceIndex, vertexIndex);
    glm::vec2 texCoord = meshPrimitive->texCoords[index];
    outUv[0] = texCoord.x;
    outUv[1] = texCoord.y;
}

void MikkTSpaceHandler::setTSpaceBasic(const SMikkTSpaceContext* context, const float* tangentU, float fSign, int faceIndex, int vertexIndex) {
    const auto meshPrimitive = reinterpret_cast<GltfMeshPrimitive*>(context->m_pUserData);
    int index = getVertexIndex(context, faceIndex, vertexIndex);
    glm::vec4 tangent(tangentU[0], tangentU[1], tangentU[2], fSign);
    meshPrimitive->tangents[index] = tangent;
}
