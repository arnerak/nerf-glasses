#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <unordered_map>

class NgpGrid {
private:
    struct MipPoint {
        uint8_t x, y, z, level;
        MipPoint(int xx, int yy, int zz, int lvl) : x(xx), y(yy), z(zz), level(lvl)
        {
        }
        bool operator==(const MipPoint& other) const {
            return x == other.x && y == other.y && z == other.z && level == other.level;
        }
    };

    struct MipPointHash {
        std::size_t operator()(const MipPoint& pnt) const {
            return (pnt.level << 24 | pnt.z << 16 | pnt.y << 8 | pnt.x) * 0x9e3779b9;
        }
    };


public:
    typedef std::unordered_set<MipPoint, MipPointHash> DensityPointSet;
    DensityPointSet density_points;

public:
    NgpGrid(const std::vector<uint8_t>& density_grid) {
        int mip_size = 128 * 128 * 128;
        for (uint8_t lvl = 0; lvl < 8; ++lvl) {
            for (uint8_t x = 0; x < 128; ++x) {
                for (uint8_t y = 0; y < 128; ++y) {
                    for (uint8_t z = 0; z < 128; ++z) {
                        if (lvl > 0 && x >= 32 && x < 96 && y >= 32 && y < 96 && z >= 32 && z < 96) continue;
                        int idx = grid_idx(x, y, z, lvl);
                        if (density_grid[idx] != 0) {
                            //density_points.insert({ x, y, z, lvl });
                            density_points.insert({ x, y, z, lvl });
                        }
                    }
                }
            }
        }
        std::cout << "num of points set in grid " << density_points.size() << std::endl;
    }

    inline int grid_idx(int x, int y, int z, int lvl)
    {
        return x + 128 * (y + 128 * (z + 128 * lvl));
    }


    std::vector<MipPoint> get_neighbors(const MipPoint& point) {
        uint8_t x = point.x;
        uint8_t y = point.y;
        uint8_t z = point.z;
        uint8_t mip = point.level;
        std::vector<MipPoint> neighbors;

        // check left and right 
        if (x - 1 >= 0 && density_points.find({ x - 1, y, z, mip }) != density_points.end())
            neighbors.push_back({ x - 1, y, z, mip });
        if (x + 1 < 128 && density_points.find({ x + 1, y, z, mip }) != density_points.end())
            neighbors.push_back({ x + 1, y, z, mip });
        // check top and bottom neighbors
        if (y - 1 >= 0 && density_points.find({ x, y - 1, z, mip }) != density_points.end())
            neighbors.push_back({ x, y - 1, z, mip });
        if (y + 1 < 128 && density_points.find({ x, y + 1, z, mip }) != density_points.end())
            neighbors.push_back({ x, y + 1, z, mip });
        // check front and back neighbors
        if (z - 1 >= 0 && density_points.find({ x, y, z - 1, mip }) != density_points.end())
            neighbors.push_back({ x, y, z - 1, mip });
        if (z + 1 < 128 && density_points.find({ x, y, z + 1, mip }) != density_points.end())
            neighbors.push_back({ x, y, z + 1, mip });

        // find neighbor at child->parent boundary
        if (mip < 7) {
            // indices in parent mip
            int mx = 32 + x / 2;
            int my = 32 + y / 2;
            int mz = 32 + z / 2;
            if (x == 0 && density_points.find({ 31, my, mz, mip + 1 }) != density_points.end())
                neighbors.push_back({ 31, my, mz, mip + 1 });
            if (x == 127 && density_points.find({ 96, my, mz, mip + 1 }) != density_points.end())
                neighbors.push_back({ 96, my, mz, mip + 1 });
            if (y == 0 && density_points.find({ mx, 31, mz, mip + 1 }) != density_points.end())
                neighbors.push_back({ mx, 31, mz, mip + 1 });
            if (y == 127 && density_points.find({ mx, 96, mz, mip + 1 }) != density_points.end())
                neighbors.push_back({ mx, 96, mz, mip + 1 });
            if (z == 0 && density_points.find({ mx, my, 31, mip + 1 }) != density_points.end())
                neighbors.push_back({ mx, my, 31, mip + 1 });
            if (z == 127 && density_points.find({ mx, my, 96, mip + 1 }) != density_points.end())
                neighbors.push_back({ mx, my, 96, mip + 1 });
        }

        // find neighbor at parent->child boundary
        if (mip > 0) {
            // indices in child mip
            uint8_t cx = x * 2 - 64;
            uint8_t cy = y * 2 - 64;
            uint8_t cz = z * 2 - 64;
            if (x == 31) {
                if (density_points.find({ 0, cy + 0, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 0, cy + 0, cz + 0, mip - 1 });
                }
                if (density_points.find({ 0, cy + 0, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 0, cy + 0, cz + 1, mip - 1 });
                }
                if (density_points.find({ 0, cy + 1, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 0, cy + 1, cz + 0, mip - 1 });
                }
                if (density_points.find({ 0, cy + 1, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 0, cy + 1, cz + 1, mip - 1 });
                }
            }
            if (x == 96) {
                if (density_points.find({ 127, cy + 0, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 127, cy + 0, cz + 0, mip - 1 });
                }
                if (density_points.find({ 127, cy + 0, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 127, cy + 0, cz + 1, mip - 1 });
                }
                if (density_points.find({ 127, cy + 1, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 127, cy + 1, cz + 0, mip - 1 });
                }
                if (density_points.find({ 127, cy + 1, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ 127, cy + 1, cz + 1, mip - 1 });
                }
            }
            if (y == 31) {
                if (density_points.find({ cx + 0, 0, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, 0, cz + 0, mip - 1 });
                }
                if (density_points.find({ cx + 0, 0, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, 0, cz + 1, mip - 1 });
                }
                if (density_points.find({ cx + 1, 0, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, 0, cz + 0, mip - 1 });
                }
                if (density_points.find({ cx + 1, 0, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, 0, cz + 1, mip - 1 });
                }
            }
            if (y == 96) {
                if (density_points.find({ cx + 0, 127, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, 127, cz + 0, mip - 1 });
                }
                if (density_points.find({ cx + 0, 127, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, 127, cz + 1, mip - 1 });
                }
                if (density_points.find({ cx + 1, 127, cz + 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, 127, cz + 0, mip - 1 });
                }
                if (density_points.find({ cx + 1, 127, cz + 1, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, 127, cz + 1, mip - 1 });
                }
            }
            if (z == 31) {
                if (density_points.find({ cx + 0, cy + 0, 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, cy + 0, 0, mip - 1 });
                }
                if (density_points.find({ cx + 0, cy + 1, 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, cy + 1, 0, mip - 1 });
                }
                if (density_points.find({ cx + 1, cy + 0, 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, cy + 0, 0, mip - 1 });
                }
                if (density_points.find({ cx + 1, cy + 1, 0, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, cy + 1, 0, mip - 1 });
                }
            }
            if (z == 96) {
                if (density_points.find({ cx + 0, cy + 0, 127, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, cy + 0, 127, mip - 1 });
                }
                if (density_points.find({ cx + 0, cy + 1, 127, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 0, cy + 1, 127, mip - 1 });
                }
                if (density_points.find({ cx + 1, cy + 0, 127, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, cy + 0, 127, mip - 1 });
                }
                if (density_points.find({ cx + 1, cy + 1, 127, mip - 1 }) != density_points.end()) {
                    neighbors.push_back({ cx + 1, cy + 1, 127, mip - 1 });
                }
            }
        }

        return neighbors;
    }

    std::vector<DensityPointSet> cluster() {
        std::vector<DensityPointSet> clusters;
        std::vector<MipPoint> noise;
        auto unvisited_points = density_points;

        while (!unvisited_points.empty()) {
            auto P = *unvisited_points.begin();
            unvisited_points.erase(unvisited_points.begin());
            auto neighbors = get_neighbors(P);

            if (!neighbors.empty()) {
                DensityPointSet C;
                C.insert(P);

                int i = 0;
                while (i < neighbors.size()) {
                    auto P = neighbors[i];
                    if (unvisited_points.find(P) != unvisited_points.end()) {
                        unvisited_points.erase(P);
                        auto N = get_neighbors(P);
                        if (!N.empty()) {
                            neighbors.insert(neighbors.end(), N.begin(), N.end());
                        }
                    }
                    C.insert(P);
                    i++;
                }
                clusters.push_back(C);
            }
            else {
                //clusters.push_back({ P });
                //noise.push_back(P);
            }
        }

        return clusters;
    }

    static void to_ngp_grid(uint8_t* grid, const DensityPointSet& point_set) {
        memset(grid, 0, 128 * 128 * 128 * 8);
        for (const auto& point : point_set) {
            uint8_t x = point.x;
            uint8_t y = point.y;
            uint8_t z = point.z;
            uint8_t mip = point.level;
            grid[x + 128 * (y + 128 * (z + 128 * mip))] = 1;
            for (int lvl = mip + 1; lvl < 8; ++lvl) {
                x = 32 + x / 2;
                y = 32 + y / 2;
                z = 32 + z / 2;
                grid[x + 128 * (y + 128 * (z + 128 * lvl))] = 1;
            }
        }
    }

    static int64_t point_set_importance(const DensityPointSet& point_set) {
        static std::unordered_map<const DensityPointSet*, int64_t> scores;
        auto it = scores.find(&point_set);
        if (it == scores.end()) {
            uint64_t score = 0;
            for (const auto& point : point_set) {
                score += 16 - powf(2, point.level);
            }
            scores.emplace(&point_set, score);
            return score;
        } else {
            return it->second;
        }
    }
};