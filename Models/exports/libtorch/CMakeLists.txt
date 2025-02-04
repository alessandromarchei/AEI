cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deploy_libtorch)

## These should be variables (TODO)
set(USE_CUDNN ON)
set(CAFFE2_USE_CUDNN ON)
set(USE_CUSPARSELT OFF)

## Find libTorch installation
find_package(Torch REQUIRED HINTS ${LIBTORCH_INSTALL_ROOT}/share/cmake/Torch/ QUIET)

## Build & Link
add_executable(deploy_libtorch main.cpp)
target_link_libraries(deploy_libtorch "${TORCH_LIBRARIES}")