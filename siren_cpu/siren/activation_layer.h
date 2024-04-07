#pragma once
#include <nn_layer.h>
#include <cmath>


class SinLayer : public NNLayer {
private:
	float w0;

public:
	SinLayer(float w0=30) : w0(w0) {};
	~SinLayer();

	Matrix forward(const Matrix& A) const;
};


SinLayer::~SinLayer() {}


Matrix SinLayer::forward(const Matrix& A) const {
	auto res = A;
    for (size_t i = 0; i < A.shape.y; ++i) {
        for (size_t j = 0; j < A.shape.x; ++j) {
            res[i * A.shape.x + j] = sin(w0 * A[i * A.shape.x + j]);
        }
    }
    return res;
}
