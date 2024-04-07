#pragma once
#include <nn_layer.h>


class LinearLayer : public NNLayer {
private:
	void initializeBiasWithZeros();
	void initializeWeightsRandomly();

public:
    Matrix W;
	Matrix b;

	LinearLayer(Shape W_shape);
	~LinearLayer();

	Matrix forward(const Matrix& A) const ;

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;
};


LinearLayer::LinearLayer(Shape W_shape) :
	W(W_shape), b(W_shape.y, 1) {
}


LinearLayer::~LinearLayer() {}


void LinearLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * 0.1;
		}
	}
}


void LinearLayer::initializeBiasWithZeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}
}


Matrix LinearLayer::forward(const Matrix& A) const {
	assert(W.shape.x == A.shape.y);
	return W * A + b;
}


int LinearLayer::getXDim() const {
	return W.shape.x;
}


int LinearLayer::getYDim() const {
	return W.shape.y;
}


Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}


Matrix LinearLayer::getBiasVector() const {
	return b;
}
