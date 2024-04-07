#pragma once

#include <siren.h>
#include <vector.h>
#include <sdf.h>


class NeuralSDF : public SDF{
public:
    NeuralSDF(Siren* mlp) : mlp(mlp) {};

    NeuralSDF(Siren* mlp, const Vector& color) : SDF(color), mlp(mlp) {};

    double ComputeSdf(const Vector &point) const override{
        Matrix input(1, 3);
        input[0] = point[0];
        input[1] = point[1];
        input[2] = point[2];
        return mlp->forward(input)[0];
    }

private:
    Siren* mlp;
};