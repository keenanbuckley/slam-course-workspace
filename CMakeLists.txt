cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(gradient_descent_example gradient_descent_example.cpp gradient_descent.cpp gradient_descent.h)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
