#pragma once


struct Shape {
	size_t x, y;
	Shape(size_t x = 1, size_t y = 1) : x(x), y(y){};
};


class Matrix {
private:
	void allocateMemory();

public:
	Shape shape;
	std::shared_ptr<float> W;

	Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(const Shape&);

	float& operator[](const int index);
	const float& operator[](const int index) const;
};


Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), W(nullptr) {
    
    allocateMemory();
}


Matrix::Matrix(const Shape& shape) :
	Matrix::Matrix(shape.x, shape.y) {
}


float& Matrix::operator[](const int index) {
	return W.get()[index];
}


const float& Matrix::operator[](const int index) const {
	return W.get()[index];
}


void Matrix::allocateMemory() {
    W = std::shared_ptr<float>(new float[shape.x * shape.y],
                                        [&](float* ptr){ delete[] ptr; });
}


inline Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
    auto n = lhs.shape.x;
    auto m = lhs.shape.y;
    auto p = rhs.shape.x;

    Shape new_shape(p, m);
    Matrix result(new_shape);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float acc_sum = 0;
            for (size_t k = 0; k < n; ++k) {
                acc_sum += lhs[i * n + k] * rhs[k * p + j];
            }
            result[i * p + j] = acc_sum;
        }
    }
    return result;
}


inline Matrix operator+(const Matrix &lhs, const Matrix &rhs) {
    Matrix result(lhs.shape);

    for (size_t i = 0; i < lhs.shape.y; ++i) {
        for (size_t j = 0; j < lhs.shape.x; ++j) {
            result[i * lhs.shape.x + j] = lhs[i * lhs.shape.x + j] + rhs[i * lhs.shape.x + j];
        }
    }
    return result;
}

