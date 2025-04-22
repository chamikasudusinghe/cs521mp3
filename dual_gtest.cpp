#include <gtest/gtest.h>
#include "dual_number.h"

TEST(DualNumberTest, BasicOps) {
    dual_number x(2.0f, 1.0f);
    dual_number y(3.0f, 0.0f);

    auto z = x * y;
    EXPECT_FLOAT_EQ(z.value(), 6.0f);
    EXPECT_FLOAT_EQ(z.dual(), 3.0f);
}

TEST(DualNumberTest, SinExpComposite) {
    dual_number x(0.5f, 1.0f);
    auto y = sin(exp(x));  // f(x) = sin(e^x), f' = cos(e^x) * e^x

    float expected_val = std::sin(std::exp(0.5f));
    float expected_der = std::cos(std::exp(0.5f)) * std::exp(0.5f);

    EXPECT_NEAR(y.value(), expected_val, 1e-4);
    EXPECT_NEAR(y.dual(), expected_der, 1e-4);
}

TEST(DualNumberTest, Sigmoid) {
    dual_number x(0.0f, 1.0f);
    auto y = sigmoid(x);  // s = 0.5, s' = 0.25

    EXPECT_NEAR(y.value(), 0.5f, 1e-5);
    EXPECT_NEAR(y.dual(), 0.25f, 1e-5);
}
