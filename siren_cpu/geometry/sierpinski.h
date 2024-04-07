#pragma once

#include <vector.h>
#include <sdf.h>
#include <vector>

class Sierpinski : public SDF{
public:
    Sierpinski(){};

    Sierpinski(Vector center, double size, const Vector& color = {0, 0, 0}) : SDF(color), center_(center), size_(size){
        vertices.push_back(Vector{1, 1, 1});
        vertices.push_back(Vector{-1, -1, 1});
        vertices.push_back(Vector{1, -1, -1});
        vertices.push_back(Vector{-1, 1, -1});
        level_ = 3;
    };


    Vector Transform2Local(const Vector &point) const {
        return (point - center_) / size_;
    }

    Vector Fold(const Vector &point, const Vector& pointOnPlane, const Vector& planeNormal) const {
        // Center plane on origin for distance calculation
        double distToPlane = DotProduct(point - pointOnPlane, planeNormal);
        
        // We only want to reflect if the dist is negative
        distToPlane = std::min(distToPlane, 0.0);
        return point - 2.0 * distToPlane * planeNormal;
    }

    double ComputeSdf(const Vector& point) const override{
        double scale = 1.0;
        Vector cur_point = Transform2Local(point);

        for (int i = 0; i < level_; i++) {
            // Scale point toward corner vertex, update scale accumulator
            cur_point -= vertices[0];
            cur_point.Scale(2.0);
            cur_point += vertices[0];
            scale *= 2;
            
            // Fold point across each plane
            for (int i = 1; i <= 3; i++) {
                // The plane is defined by:
                // Point on plane: The vertex that we are reflecting across
                // Plane normal: The direction from said vertex to the corner vertex
                Vector normal = vertices[0] - vertices[i];
                normal.Normalize(); 
                cur_point = Fold(cur_point, vertices[i], normal);
            }
        }
        // Now that the space has been distorted by the IFS,
        // just return the distance to a tetrahedron
        // Divide by scale accumulator to correct the distance field
        auto len =  (std::max(std::abs(cur_point[0] + cur_point[1]) - cur_point[2],
            std::abs(cur_point[0] - cur_point[1]) + cur_point[2]) - 1.0) / std::sqrt(3.);
        return size_ * len / scale;
    }

private:
    Vector center_;
    double size_;

    std::vector<Vector> vertices;
    int level_;
};
