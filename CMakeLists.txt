cmake_minimum_required(VERSION 3.14)
project(MP3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(dual_test dual_test.cpp)
add_executable(dual_gtest dual_gtest.cpp)

add_subdirectory(external/googletest)
include_directories(external/googletest/googletest/include)

target_link_libraries(dual_gtest gtest gtest_main pthread)
