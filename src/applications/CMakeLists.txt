cmake_minimum_required(VERSION 3.0)

add_executable(local local.cpp)
target_include_directories(local
  PUBLIC
  ${RDMA_ML_SOURCE_DIR})
target_link_libraries(local RDMA_ML utils)

add_executable(worker worker.cpp worker.hpp)
target_include_directories(worker
  PUBLIC
  ${RDMA_ML_SOURCE_DIR})
target_link_libraries(worker coordinator RDMA_ML utils)

add_executable(server server.cpp server.hpp)
target_include_directories(server
  PUBLIC
  ${RDMA_ML_SOURCE_DIR})
target_link_libraries(server coordinator RDMA_ML utils)
