#pragma once

#include <vector.h>

class Ray {
public:
    Ray(Vector origin, Vector direction) : origin_(origin), direction_(direction){};
    const Vector& GetOrigin() const {
        return origin_;
    };
    const Vector& GetDirection() const {
        return direction_;
    };
    void ShiftOrigin(double t) {
        origin_ += t * direction_;
    }

private:
    Vector origin_;
    Vector direction_;
};
