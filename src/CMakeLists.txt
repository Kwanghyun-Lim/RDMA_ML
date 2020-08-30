cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

PROJECT(RDMA_ML CXX)

add_subdirectory(applications)
add_subdirectory(coordinator)
add_subdirectory(utils)

add_library(RDMA_ML
  SHARED
  ml_model.hpp
  ml_model.cpp
  log_reg.hpp
  log_reg.cpp
  dnn.hpp
  dnn.cpp)
target_link_libraries(RDMA_ML PUBLIC openblas utils)
