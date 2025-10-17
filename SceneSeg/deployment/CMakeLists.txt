cmake_minimum_required(VERSION 3.16)
project(sceneseg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT root (adjust if needed or pass via -DTENSORRT_ROOT=/path)
if (NOT DEFINED TENSORRT_ROOT)
    # Common default locations
    set(TENSORRT_ROOT " ~/TensorRT-10.4.0.26")
endif()

# CUDA / CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED core imgproc imgcodecs videoio)

# TensorRT headers & libs
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include ${TENSORRT_ROOT}/include/x86_64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
# (Optional) parsers if you ever need them:
# find_library(NVONNXPARSER_LIBRARY nvonnxparser HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "Could not find TensorRT. Set TENSORRT_ROOT or adjust search paths.")
endif()


message("EXECUTABLE IS CALLED sceneseg")
add_executable(sceneseg deploy.cpp)
target_include_directories(sceneseg PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(sceneseg PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)

# For TensorRT 8.x symbols on some distros
if (UNIX AND NOT APPLE)
    target_link_libraries(sceneseg PRIVATE dl pthread)
endif()

# Faster symbols / LTO (optional)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if (ipo_supported)
    set_property(TARGET sceneseg PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
