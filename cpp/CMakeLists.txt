cmake_minimum_required(VERSION 3.14)
project(CorruptDetectCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(corrupt_detect_infer main.cpp)
target_link_libraries(corrupt_detect_infer ${OpenCV_LIBS} onnxruntime)

set_target_properties(corrupt_detect_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
