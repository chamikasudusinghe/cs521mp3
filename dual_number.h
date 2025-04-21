// dual_number.h
#ifndef DUAL_NUMBER_H
#define DUAL_NUMBER_H

#include <cmath>
#include <iostream>

class dual_number {
private:
    float val, der;

public:
    // Constructors
    dual_number() : val(0.0f), der(0.0f) {}
    dual_number(float v) : val(v), der(0.0f) {}
    dual_number(float v, float d) : val(v), der(d) {}

    // Accessors
    float value() const { return val; }
    float dual() const { return der; }

    // Operator overloads
    dual_number operator+(const dual_number& other) const {
        return dual_number(val + other.val, der + other.der);
    }

    dual_number operator-(const dual_number& other) const {
        return dual_number(val - other.val, der - other.der);
    }

    dual_number operator*(const dual_number& other) const {
        return dual_number(val * other.val, val * other.der + der * other.val);
    }

    dual_number operator/(const dual_number& other) const {
        float denom = other.val * other.val;
        return dual_number(val / other.val, (der * other.val - val * other.der) / denom);
    }

    // Math functions
    friend dual_number sin(const dual_number& x) {
        return dual_number(std::sin(x.val), std::cos(x.val) * x.der);
    }

    friend dual_number cos(const dual_number& x) {
        return dual_number(std::cos(x.val), -std::sin(x.val) * x.der);
    }

    friend dual_number exp(const dual_number& x) {
        float e = std::exp(x.val);
        return dual_number(e, e * x.der);
    }

    friend dual_number ln(const dual_number& x) {
        return dual_number(std::log(x.val), x.der / x.val);
    }

    friend dual_number relu(const dual_number& x) {
        return x.val > 0 ? x : dual_number(0.0f, 0.0f);
    }

    friend dual_number sigmoid(const dual_number& x) {
        float s = 1.0f / (1.0f + std::exp(-x.val));
        return dual_number(s, s * (1 - s) * x.der);
    }

    friend dual_number tanh(const dual_number& x) {
        float t = std::tanh(x.val);
        return dual_number(t, (1 - t * t) * x.der);
    }
};

#endif // DUAL_NUMBER_H