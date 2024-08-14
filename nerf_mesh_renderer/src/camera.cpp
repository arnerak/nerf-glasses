#include "camera.h"

#include "common.h"

Camera::Camera(glm::vec3 eye, glm::vec3 center, glm::vec3 up, float fovY, float aspectRatio)
        : _eye(eye), _center(center), _up(up), _viewMatrix(glm::lookAt(eye, center, up)),
          _fovY(glm::radians(fovY)), _aspectRatio(aspectRatio),
          _u(glm::vec3(1.0f)),
          _v(glm::vec3(1.0f)),
          _w(glm::vec3(1.0f))
{
    // Properly set (u, v, w).
    setLocalCoordinateFrame();
}

void Camera::move(const float angleX, const float angleY) {
    glm::mat4 rotX = glm::rotate(glm::mat4(1.0f), glm::radians(angleX * SENSITIVITY_ROTATION), _up);
    glm::mat4 rotY = glm::rotate(glm::mat4(1.0f), glm::radians(angleY * SENSITIVITY_ROTATION), _u);

    _viewMatrix = _viewMatrix * rotY * rotX;
    _eye = glm::inverse(_viewMatrix)[3];

    setLocalCoordinateFrame();
}

void Camera::zoom(const float zoomDir) {
    const glm::vec3 centerToEye = _eye - _center;

    float zoom = (zoomDir > 0) ? SENSITIVITY_ZOOM : 1.0f / SENSITIVITY_ZOOM;
    zoom = glm::clamp(glm::length(centerToEye) * zoom, 0.1f, 50.0f);

    _eye = _center + glm::normalize(centerToEye) * zoom;
    _viewMatrix = glm::lookAt(_eye, _center, _up);

    setLocalCoordinateFrame();
}

glm::vec3 Camera::eye() const {
    return _eye;
}

glm::vec3 Camera::u() const {
    return _u;
}

glm::vec3 Camera::v() const {
    return _v;
}

glm::vec3 Camera::w() const {
    return _w;
}

float Camera::fovY() const {
    return glm::degrees(_fovY);
}

void Camera::setLocalCoordinateFrame() {
    float wLength = glm::length(_center - _eye);
    float vLength = wLength * tanf(0.5f * _fovY);
    float uLength = vLength * _aspectRatio;

    glm::mat4 transposedViewMatrix = glm::transpose(_viewMatrix);
    _u = glm::vec3(transposedViewMatrix[0] * uLength);
    _v = glm::vec3(transposedViewMatrix[1] * vLength);
    _w = -glm::vec3(transposedViewMatrix[2] * wLength);
}