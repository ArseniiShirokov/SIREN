#pragma once

#include <vector.h>

class SDF {
public:
    SDF() : color(Vector{1, 1, 1}) {};
    SDF(const Vector& color) : color(color) {};

    virtual double ComputeSdf(const Vector &point) const = 0;

    Vector GetColor() const {
        return color;
    }

protected:
    Vector color;
};


Vector EstimateNormal(const SDF* obj, const Vector& z, double eps = 1e-4) {
    Vector z1 = z + Vector{eps, 0, 0};
    Vector z2 = z - Vector{eps, 0, 0};
    Vector z3 = z + Vector{0, eps, 0};
    Vector z4 = z - Vector{0, eps, 0};
    Vector z5 = z + Vector{0, 0, eps};
    Vector z6 = z - Vector{0, 0, eps};
    double dx = obj->ComputeSdf(z1) - obj->ComputeSdf(z2);
    double dy = obj->ComputeSdf(z3) - obj->ComputeSdf(z4);
    double dz = obj->ComputeSdf(z5) - obj->ComputeSdf(z6);
    auto v = Vector{dx, dy, dz};
    return v / (2.0 * eps);
}