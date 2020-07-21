cmake_minimum_required(VERSION 3.0)

add_executable(worker worker.cpp worker.hpp)
target_include_directories(worker
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(worker coordinator utils)

add_executable(server server.cpp)
target_include_directories(server
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(server coordinator logistic_regression utils)
