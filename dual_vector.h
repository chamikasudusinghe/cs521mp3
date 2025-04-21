// dual_vector.h
#ifndef DUAL_VECTOR_H
#define DUAL_VECTOR_H

#include <vector>
#include "dual_number.h"

class dual_vector {
private:
    std::vector<dual_number> data;

public:
    dual_vector(std::size_t n) : data(n) {}
    dual_vector(const std::vector<dual_number>& vec) : data(vec) {}

    std::size_t size() const { return data.size(); }

    dual_number& operator[](std::size_t i) { return data[i]; }
    const dual_number& operator[](std::size_t i) const { return data[i]; }

    std::vector<dual_number> get_data() const { return data; }

    dual_vector apply(dual_number(*func)(const dual_number&)) const {
        std::vector<dual_number> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = func(data[i]);
        }
        return dual_vector(result);
    }
};

#endif // DUAL_VECTOR_H
