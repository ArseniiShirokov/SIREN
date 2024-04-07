#pragma once

#include <matrix.h>

class NNLayer {

public:
	virtual ~NNLayer() = 0;
	virtual Matrix forward(const Matrix& A) const = 0;
};

inline NNLayer::~NNLayer() {}