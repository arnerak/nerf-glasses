/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   bounding_box.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 *  @brief  CUDA/C++ AABB implementation.
 */

#pragma once

#include "ngp_common.cuh"

NGP_NAMESPACE_BEGIN

struct BoundingBox {
    Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());

    // Min and max w/o translation applied.
    // Probably better to provide getters for min/max which return the values with translation applied.
    // But this way we can keep the refactoring to this file only, which is good enough for the scope of the thesis.
    Eigen::Vector3f untransformedMin = min;
    Eigen::Vector3f untransformedMax = max;

    Eigen::Vector3f translation = Eigen::Vector3f::Zero();

    HOST_DEVICE void setUntransformedMinMax() {
        untransformedMin = min;
        untransformedMax = max;
    }

    HOST_DEVICE BoundingBox() = default;

    HOST_DEVICE BoundingBox(const Eigen::Vector3f& a, const Eigen::Vector3f&  b): min{a}, max{b}
    {
        setUntransformedMinMax();
    }

    HOST_DEVICE BoundingBox(const BoundingBox& other) : min{other.min}, max{other.max}, translation{other.translation}
    {
        setUntransformedMinMax();
    }

    HOST_DEVICE BoundingBox& operator=(const BoundingBox& other) {
        min = other.min;
        max = other.max;
        translation = other.translation;

        setUntransformedMinMax();

        return *this;
    }

    HOST_DEVICE void enlarge(const BoundingBox& other) {
        min = min.cwiseMin(other.min);
        max = max.cwiseMax(other.max);

        setUntransformedMinMax();
    }

    HOST_DEVICE void enlarge(const Eigen::Vector3f& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);

        setUntransformedMinMax();
    }

    HOST_DEVICE void inflate(float amount) {
        min -= Eigen::Vector3f::Constant(amount);
        max += Eigen::Vector3f::Constant(amount);

        setUntransformedMinMax();
    }

    HOST_DEVICE Eigen::Vector3f diag() const {
        return max - min;
    }

    HOST_DEVICE Eigen::Vector3f relative_pos(const Eigen::Vector3f& pos) const {
        return (pos - min).cwiseQuotient(diag());
    }

    HOST_DEVICE Eigen::Vector3f center() const {
        return 0.5f * (max + min);
    }

    HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
        BoundingBox result = *this;
        result.min = result.min.cwiseMax(other.min);
        result.max = result.max.cwiseMin(other.max);
        result.setUntransformedMinMax();
        return result;
    }

    HOST_DEVICE bool intersects(const BoundingBox& other) const {
        return !intersection(other).is_empty();
    }

    HOST_DEVICE Eigen::Vector2f ray_intersect(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir) const {
        const auto posT = pos - translation;

        float tmin = (min.x() - posT.x()) / dir.x();
        float tmax = (max.x() - posT.x()) / dir.x();

        if (tmin > tmax) {
            tcnn::host_device_swap(tmin, tmax);
        }

        float tymin = (min.y() - posT.y()) / dir.y();
        float tymax = (max.y() - posT.y()) / dir.y();

        if (tymin > tymax) {
            tcnn::host_device_swap(tymin, tymax);
        }

        if (tmin > tymax || tymin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tymin > tmin) {
            tmin = tymin;
        }

        if (tymax < tmax) {
            tmax = tymax;
        }

        float tzmin = (min.z() - posT.z()) / dir.z();
        float tzmax = (max.z() - posT.z()) / dir.z();

        if (tzmin > tzmax) {
            tcnn::host_device_swap(tzmin, tzmax);
        }

        if (tmin > tzmax || tzmin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tzmin > tmin) {
            tmin = tzmin;
        }

        if (tzmax < tmax) {
            tmax = tzmax;
        }

        return { tmin, tmax };
    }

    HOST_DEVICE bool is_empty() const {
        return (max.array() < min.array()).any();
    }

    HOST_DEVICE bool contains(const Eigen::Vector3f& p) const {
        const auto pT = p - translation;
        return
                pT.x() >= min.x() && pT.x() <= max.x() &&
                pT.y() >= min.y() && pT.y() <= max.y() &&
                pT.z() >= min.z() && pT.z() <= max.z();
    }

    void translate() {
        min = untransformedMin - translation;
        max = untransformedMax - translation;
    }
};

inline std::ostream& operator<<(std::ostream& os, const BoundingBox& bb) {
    os << "[";
    os << "min=[" << bb.min.x() << "," << bb.min.y() << "," << bb.min.z() << "], ";
    os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "]";
    os << "]";
    return os;
}

NGP_NAMESPACE_END