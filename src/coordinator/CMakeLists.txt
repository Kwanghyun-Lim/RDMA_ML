cmake_minimum_required(VERSION 3.0)

add_library(coordinator
  SHARED
  tcp.hpp tcp.cpp
  sst.h sst_impl.h
  verbs.h verbs.cpp
  poll_utils.h poll_utils.cpp
  connection_manager.h connection_manager.cpp
  ml_sst.hpp ml_sst.cpp)
target_include_directories(coordinator PUBLIC ${RDMA_ML_SOURCE_DIR})
target_link_libraries(coordinator RDMA_ML utils ibverbs rdmacm pthread)
