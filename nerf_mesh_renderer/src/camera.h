#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

class Camera {
public:
    /**
     *
     * @param eye The position of the camera.
     * @param center Where the camera looks at.
     * @param up The up direction of the camera.
     * @param fovY Vertical field of view in degrees (internally converted to radians).
     * @param aspectRatio The aspect ratio.
     */
    Camera(glm::vec3 eye, glm::vec3 center, glm::vec3 up, float fovY, float aspectRatio);

    void move(float angleX, float angleY);
    void zoom(float zoomDir);

    [[nodiscard]] glm::vec3 eye() const;
    [[nodiscard]] glm::vec3 u() const;
    [[nodiscard]] glm::vec3 v() const;
    [[nodiscard]] glm::vec3 w() const;
    [[nodiscard]] float fovY() const;

private:
    static constexpr float SENSITIVITY_ROTATION = 0.1f;
    static constexpr float SENSITIVITY_ZOOM = 1.1f;

    glm::vec3 _eye;
    glm::vec3 _center;
    glm::vec3 _up;
    glm::mat4 _viewMatrix;

    float _fovY;
    float _aspectRatio;

    glm::vec3 _u;
    glm::vec3 _v;
    glm::vec3 _w;

    void setLocalCoordinateFrame();
};
