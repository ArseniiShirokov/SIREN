#include <camera_options.h>
#include <ray.h>
#include <math.h>

inline std::vector<Vector> ComputeRotationMat(Vector forward);
inline Vector Rotate(const Vector& base, const std::vector<Vector>& mat);

class Transformer {
    std::vector<Vector> rotation_mat_;
    Vector new_origin_;
    double width_;
    double height_;
    double step_;

public:
    Transformer(const CameraOptions& options) {
        height_ = 2 * tan(options.fov / 2);
        width_ = (height_ * options.screen_width) / static_cast<double>(options.screen_height);
        new_origin_ = options.look_from;
        rotation_mat_ = ComputeRotationMat(options.look_from - options.look_to);
        step_ = width_ / options.screen_width;
    };

    Ray MakeRay(int y, int x) {
        Vector default_dir = {-width_ / 2 + (x + 0.5) * step_, height_ / 2 - (y + 0.5) * step_, -1};
        default_dir.Normalize();
        Vector rotated_dir = Rotate(default_dir, rotation_mat_);
        return Ray(new_origin_, rotated_dir);
    }
};

inline Vector Rotate(const Vector& base, const std::vector<Vector>& mat) {
    Vector result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += mat[j][i] * base[j];
        }
    }
    result.Normalize();
    return result;
}

inline std::vector<Vector> ComputeRotationMat(Vector forward) {
    Vector tmp(std::initializer_list<double>{0, 1, 0});
    forward.Normalize();
    if (std::fabs(DotProduct(forward, tmp)) == 1) {
        tmp = {0, 0, -forward[1]};
    }
    auto right = CrossProduct(tmp, forward);
    auto up = CrossProduct(forward, right);
    right.Normalize();
    up.Normalize();
    return std::vector<Vector>{right, up, forward};
}