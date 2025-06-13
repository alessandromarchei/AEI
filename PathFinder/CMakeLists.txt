cmake_minimum_required(VERSION 3.10)
project(PathFinder)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# Include OpenCV and your own headers
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Add source files
add_executable(${PROJECT_NAME}
    src/path_finder.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} yaml-cpp)

# Copy YAML config to build directory
configure_file(test/image_to_world_transform.yaml image_to_world_transform.yaml COPYONLY)
configure_file(test/test.yaml test.yaml COPYONLY)