cmake_minimum_required(VERSION 3.0)

add_executable(local_dnn local_dnn.cpp)
target_include_directories(local_dnn
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(local_dnn logistic_regression utils)

add_executable(worker worker.cpp worker.hpp)
target_include_directories(worker
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(worker coordinator logistic_regression utils)

add_executable(server server.cpp server.hpp)
target_include_directories(server
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(server coordinator logistic_regression utils)
