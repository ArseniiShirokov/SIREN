#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <algorithm>

class Vector {
public:
    Vector() : data_({0, 0, 0}){};
    Vector(std::initializer_list<double> list) {
        int i = 0;
        for (auto& c : list) {
            data_[i++] = c;
        }
    };
    Vector(std::array<double, 3> data) : data_(data){};

    double& operator[](size_t ind) {
        return data_[ind];
    };
    double operator[](size_t ind) const {
        return data_[ind];
    };

    bool operator==(const Vector& gt) {
        return ((*this)[0] == gt[0]) && ((*this)[1] == gt[1]) && ((*this)[2] == gt[2]);
    }

    Vector& operator-=(const Vector& rhs) {
        for (size_t i = 0; i < (*this).Size(); ++i) {
            (*this)[i] = (*this)[i] - rhs[i];
        }
        return *this;
    }
    Vector& operator+=(const Vector& rhs) {
        for (size_t i = 0; i < (*this).Size(); ++i) {
            (*this)[i] += rhs[i];
        }
        return *this;
    }

    Vector operator-() const {
        std::array<double, 3> ans;
        for (size_t i = 0; i < (*this).Size(); ++i) {
            ans[i] = -(*this)[i];
        }
        return Vector(ans);
    }

    void Normalize();

    void Scale(double value) {
        for (size_t i = 0; i < (*this).Size(); ++i) {
            (*this)[i] *= value;
        }
    }

    size_t Size() const {
        return data_.size();
    }

private:
    std::array<double, 3> data_;
};

inline Vector operator-(Vector lhs, const Vector& rhs) {
    return lhs -= rhs;
}

inline Vector operator+(Vector lhs, const Vector& rhs) {
    return lhs += rhs;
}

inline Vector operator*(Vector lhs, double value) {
    std::array<double, 3> ans;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans[i] = lhs[i] * value;
    }
    return Vector(ans);
}

inline Vector operator*(double value, Vector rhs) {
    return rhs * value;
}

inline Vector operator/(Vector rhs, double value) {
    return rhs * (1 / value);
}

inline Vector operator+(double value, Vector rhs) {
    return rhs += Vector({value, value, value});
}

inline Vector operator*(const Vector& lhs, const Vector& rhs) {
    std::array<double, 3> ans;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans[i] = lhs[i] * rhs[i];
    }
    return Vector(ans);
}

inline Vector operator/(const Vector& lhs, const Vector& rhs) {
    std::array<double, 3> ans;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans[i] = lhs[i] / rhs[i];
    }
    return Vector(ans);
}

inline double DotProduct(const Vector& lhs, const Vector& rhs) {
    double ans = 0.0;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans += lhs[i] * rhs[i];
    }
    return ans;
}

inline Vector CrossProduct(const Vector& a, const Vector& b) {
    return Vector(
        {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]});
}

inline double Length(const Vector& vec) {
    return std::sqrt(DotProduct(vec, vec));
}

void Vector::Normalize() {
    Scale(1.0 / Length(*this));
}
