#pragma once

#include <vector.h>
#include <sdf.h>

class UnionSDF : public SDF{
public:
    UnionSDF(){};
    UnionSDF(std::vector<SDF*> objects) : objects_(std::move(objects)){};
    UnionSDF(std::vector<SDF*> objects, const Vector& color) : SDF(color), objects_(std::move(objects)){};

    double ComputeSdf(const Vector &point) const override{
        double dist = 1e+5;
        for (auto& obj : objects_) {
            dist = std::min(dist, obj->ComputeSdf(point));
        }
        return dist;
    }

    const SDF* GetHittedObject(const Vector &point) const {
        double min_dist = 1e+5;
        SDF* hitted_object;

        for (auto& obj : objects_) {
            double dist = obj->ComputeSdf(point);
            if (dist < min_dist) {
                min_dist = dist;
                hitted_object = obj;
            }
        }
        return hitted_object;
    }

private:
    std::vector<SDF*> objects_;
};
