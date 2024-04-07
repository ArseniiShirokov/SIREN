#pragma once

#include <vector.h>
#include <sdf.h>

class Sphere : public SDF{
public:
    Sphere(){};
    Sphere(Vector center, double radius) : center_(center), radius_(radius){};
    Sphere(Vector center, double radius, const Vector& color) : SDF(color), center_(center), radius_(radius){};

    const Vector& GetCenter() const {
        return center_;
    }
    double GetRadius() const {
        return radius_;
    }
    double ComputeSdf(const Vector &point) const override{
        return Length(point - center_) - radius_;
    }

private:
    Vector center_;
    double radius_;
};
