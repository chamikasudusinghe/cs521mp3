#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "dual_number.h"

#ifndef FUNC
#define FUNC SIN
#endif

float scalar_func(float x) {

    #if defined(SIN)
        return std::sin(x);
    #elif defined(EXP2)
        return std::exp(x * x);
    #elif defined(LOGTANH)
        return std::log(x + 1.0f) + std::tanh(x);
    #elif defined(SIGEXP)
        return 1.0f / (1.0f + std::exp(-std::exp(x)));
    #elif defined(XCOMBO)
        return x * std::sin(x) + x * std::exp(x);
    #else
        return x;
    #endif

}

dual_number dual_func(dual_number x) {

    #if defined(SIN)
        return sin(x);
    #elif defined(EXP2)
        return exp(x * x);
    #elif defined(LOGTANH)
        return ln(x + dual_number(1.0f)) + tanh(x);
    #elif defined(SIGEXP)
        return sigmoid(exp(x));
    #elif defined(XCOMBO)
        return x * sin(x) + x * exp(x);
    #else
        return x;
    #endif

}

int main(int argc, char** argv) {
    int n = 1000;
    if (argc > 1) n = std::stoi(argv[1]);

    auto start = std::chrono::high_resolution_clock::now();

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float x = i * 0.001f;
        #ifdef PLAIN_ONLY
                sum += scalar_func(x);
        #else
                dual_number dx(x, 1.0f);
                sum += dual_func(dx).value();
        #endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    volatile float sink = sum;
    std::cout << us / 1000.0 << std::endl; // in ms

    return 0;
}